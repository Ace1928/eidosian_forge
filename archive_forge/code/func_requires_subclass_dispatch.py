from typing import Any, List, Optional, Tuple, Union
import torch.utils._pytree as pytree
from torch import Tensor
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from .schemas import SubclassCreationMeta, ViewAndMutationMeta
from .utils import strict_zip
def requires_subclass_dispatch(args, fw_metadata: ViewAndMutationMeta) -> bool:
    args_flattened = pytree.arg_tree_leaves(*args)
    any_subclass_args = any((is_traceable_wrapper_subclass(x) for x in args_flattened if isinstance(x, Tensor)))
    any_subclass_outputs = any((is_traceable_wrapper_subclass(x) for x in fw_metadata.traced_tangents if isinstance(x, Tensor)))
    return any_subclass_args or any_subclass_outputs