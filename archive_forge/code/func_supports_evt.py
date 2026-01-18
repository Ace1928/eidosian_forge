import copy
import logging
import re
from typing import cast, Dict, List, Optional, Tuple
from ...config import cuda as inductor_cuda_config
from ...ir import Buffer, CUDATemplateBuffer, FixedLayout, IRNode, Layout
from ..common import IndentedBuffer
from . import cutlass_utils
from .cuda_kernel import CUDATemplateKernel
from .cuda_template import CUTLASSTemplate
from .cutlass_epilogue_gen import (
@staticmethod
def supports_evt(op: 'cutlass_library.gemm_op.GemmOperation') -> bool:
    """
        returns True if the op is capable of flexible epilogue fusions
        using epilogue visitor trees.

        See https://github.com/NVIDIA/cutlass/blob/e01b9b5029b7caca5a43c29f7d2714d7cf1dcae8/examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu#L283-L285 # noqa: B950
        """
    assert cutlass_utils.try_import_cutlass()
    import cutlass_library.library as cutlass_lib
    if op.gemm_kind != cutlass_lib.GemmKind.Universal3x:
        return False
    if op.epilogue_schedule not in (cutlass_lib.EpilogueScheduleType.TmaWarpSpecialized, cutlass_lib.EpilogueScheduleType.TmaWarpSpecializedCooperative):
        return False
    return True