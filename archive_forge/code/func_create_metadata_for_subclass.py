from typing import Any, List, Optional, Tuple, Union
import torch.utils._pytree as pytree
from torch import Tensor
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from .schemas import SubclassCreationMeta, ViewAndMutationMeta
from .utils import strict_zip
def create_metadata_for_subclass(meta: ViewAndMutationMeta) -> ViewAndMutationMeta:
    input_info = []
    for inp, subclass_meta in zip(meta.input_info, meta.subclass_inp_meta):
        num_inps = 1 if isinstance(subclass_meta, int) else subclass_meta.arg_count
        for _ in range(num_inps):
            input_info.append(inp)
    output_info = []
    subclass_out_meta_user_outs_only = meta.subclass_fw_graph_out_meta[meta.num_mutated_inp_runtime_indices:]
    if meta.num_intermediate_bases > 0:
        subclass_out_meta_user_outs_only = subclass_out_meta_user_outs_only[:-meta.num_intermediate_bases]
    assert len(meta.output_info) == len(subclass_out_meta_user_outs_only)
    for out, subclass_meta in zip(meta.output_info, subclass_out_meta_user_outs_only):
        num_outs = 1 if isinstance(subclass_meta, int) else subclass_meta.arg_count
        for _ in range(num_outs):
            output_info.append(out)
    num_intermediate_bases = None
    keep_input_mutations = meta.keep_input_mutations
    traced_tangents = None
    subclass_inp_meta = None
    subclass_fw_graph_out_meta = None
    subclass_tangent_meta = None
    metadata = ViewAndMutationMeta(input_info=input_info, output_info=output_info, num_intermediate_bases=num_intermediate_bases, keep_input_mutations=keep_input_mutations, traced_tangents=traced_tangents, subclass_inp_meta=subclass_inp_meta, subclass_fw_graph_out_meta=subclass_fw_graph_out_meta, subclass_tangent_meta=subclass_tangent_meta)
    return metadata