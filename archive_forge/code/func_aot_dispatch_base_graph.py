from typing import Any, Callable, List, Optional, Tuple, Union
import torch
import torch.utils._pytree as pytree
import torch.utils.dlpack
from torch import Tensor
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import lazy_format_graph_code
from torch._logging import getArtifactLogger
from torch.fx.experimental.proxy_tensor import make_fx
from .functional_utils import assert_functional_graph
from .schemas import AOTConfig, SubclassMeta, ViewAndMutationMeta
from .traced_function_transforms import (
def aot_dispatch_base_graph(flat_fn, flat_args: List[Tensor], aot_config: AOTConfig, *, fw_metadata: ViewAndMutationMeta) -> Union[Callable, Tuple[Callable, List[Any], Optional[SubclassMeta]]]:
    fn_to_trace = fn_input_mutations_to_outputs(flat_fn, fw_metadata, keep_data_input_mutations=aot_config.keep_inference_input_mutations)
    fn_to_trace, updated_flat_args = create_functionalized_fn(fn_to_trace, flat_args, meta=fw_metadata, aot_config=aot_config, trace_joint=False)
    fn_to_trace, updated_flat_args_subclasses_desugared, maybe_subclass_meta = aot_dispatch_subclass(fn_to_trace, updated_flat_args, is_joint_structure=False, meta=fw_metadata, fw_only=flat_fn)
    fw_module = _create_graph(fn_to_trace, updated_flat_args_subclasses_desugared, aot_config=aot_config)
    copy_count = assert_functional_graph(fw_module.graph, allow_input_mutations=aot_config.keep_inference_input_mutations)
    fw_module.graph.eliminate_dead_code()
    fw_module.recompile()
    copy_count2 = assert_functional_graph(fw_module.graph, allow_input_mutations=aot_config.keep_inference_input_mutations)
    assert copy_count == copy_count2
    if aot_config.enable_log:
        aot_graphs_log.info('%s', lazy_format_graph_code('Forward graph', fw_module, aot_config.aot_id))
    if aot_config.is_export:
        assert maybe_subclass_meta is None, 'aot_export_module does not support tensor subclass inputs for now.'
        return fw_module
    return (fw_module, list(updated_flat_args_subclasses_desugared), maybe_subclass_meta)