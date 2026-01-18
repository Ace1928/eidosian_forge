from __future__ import annotations
import functools
import hashlib
import json
import os
import re
from collections import namedtuple
from pathlib import Path
from typing import Any
from dataclasses import dataclass
from .._C.libtriton.triton import (ClusterInfo, TMAInfos, add_external_libs, compile_ptx_to_cubin, get_env_vars,
from ..common.backend import get_backend, get_cuda_version_key, path_to_ptxas
from ..common.build import is_hip
from ..runtime.autotuner import OutOfResources
from ..runtime.cache import get_cache_manager, get_dump_manager, get_override_manager
from ..runtime.driver import driver
from ..runtime.jit import (JITFunction, get_cuda_stream, get_current_device, get_device_capability)
from ..tools.disasm import get_sass
from .code_generator import ast_to_ttir
from .make_launcher import make_stub
from .utils import (InfoFromBackendForTensorMap, TensorMapManager, get_ids_of_tensormaps, parse_tma_info)
def optimize_ttgir(mod, num_stages, num_warps, num_ctas, target, cluster_info, enable_warp_specialization, enable_persistent, optimize_epilogue):
    is_cuda = _is_cuda(target)
    if is_cuda:
        capability = target.capability
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    pm.add_tritongpu_coalesce_pass()
    pm.add_plan_cta_pass(cluster_info)
    if is_cuda:
        pm.add_tritongpu_rewrite_tensor_pointer_pass(capability)
        pm.add_plan_cta_pass(cluster_info)
    pm.add_tritongpu_remove_layout_conversions_pass()
    if is_cuda:
        pm.add_tritongpu_accelerate_matmul_pass(capability)
    pm.add_tritongpu_remove_layout_conversions_pass()
    if optimize_epilogue:
        pm.add_tritongpu_optimize_epilogue_pass()
    pm.add_tritongpu_optimize_dot_operands_pass()
    pm.add_cse_pass()
    ws_enabled = False
    if capability // 10 >= 9 and enable_warp_specialization and (num_warps == 4):
        pm.add_tritongpu_ws_feasibility_checking_pass(capability)
        pm.run(mod)
        ws_enabled = ir.is_ws_supported(mod)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
    if ws_enabled:
        pm.add_tritongpu_wsdecomposing_pass(capability)
        pm.add_tritongpu_wspipeline_pass(num_stages, num_warps, capability)
        pm.add_tritongpu_wsmutex_pass(capability)
        pm.add_tritongpu_wsmaterialization_pass(capability)
        pm.add_licm_pass()
        pm.add_cse_pass()
    else:
        pm.add_tritongpu_pipeline_pass(num_stages, num_warps, num_ctas, capability)
    pm.add_tritongpu_materialize_load_store_pass(num_warps, capability)
    if capability // 10 <= 8:
        pm.add_tritongpu_prefetch_pass()
    pm.add_tritongpu_optimize_dot_operands_pass()
    pm.add_tritongpu_remove_layout_conversions_pass()
    pm.add_tritongpu_decompose_conversions_pass()
    pm.add_tritongpu_ws_fixup_missing_attrs_pass()
    pm.add_tritongpu_reorder_instructions_pass()
    pm.add_cse_pass()
    pm.add_symbol_dce_pass()
    if capability // 10 >= 9:
        pm.add_tritongpu_fence_insertion_pass()
    pm.add_tritongpu_ws_fixup_missing_attrs_pass()
    pm.add_tritongpu_optimize_thread_locality_pass()
    pm.add_canonicalizer_pass()
    pm.run(mod)
    return mod