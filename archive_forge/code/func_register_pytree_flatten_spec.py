from collections import namedtuple
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Type, Optional
from torch.utils._pytree import LeafSpec, PyTree, TreeSpec
def register_pytree_flatten_spec(cls: Type[Any], flatten_fn_spec: FlattenFuncSpec, flatten_fn_exact_match_spec: Optional[FlattenFuncExactMatchSpec]=None) -> None:
    SUPPORTED_NODES[cls] = flatten_fn_spec
    SUPPORTED_NODES_EXACT_MATCH[cls] = flatten_fn_exact_match_spec