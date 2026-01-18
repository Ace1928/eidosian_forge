from typing import TYPE_CHECKING, Union, Callable, Any, Tuple, List, Optional, Dict, Set
from ._compatibility import compatibility
from .immutable_collections import immutable_dict, immutable_list
import torch
import builtins
import types
import inspect
import warnings
from torch.fx.operator_schemas import normalize_function, normalize_module, ArgsKwargsPair
from .._ops import ops as _ops
@compatibility(is_backward_compatible=True)
def replace_all_uses_with(self, replace_with: 'Node', delete_user_cb: Callable[['Node'], bool]=lambda user: True, *, propagate_meta=False) -> List['Node']:
    """
        Replace all uses of ``self`` in the Graph with the Node ``replace_with``.

        Args:

            replace_with (Node): The node to replace all uses of ``self`` with.
            delete_user_cb (Callable): Callback that is called to determine
              whether a given user of the self node should be removed.
            propagate_meta (bool): Whether or not to copy all properties
              on the .meta field of the original node onto the replacement node.
              For safety, this is only valid to do if the replacement node
              doesn't already have an existing .meta field.

        Returns:

            The list of Nodes on which this change was made.
        """
    if propagate_meta:
        assert len(replace_with.meta) == 0, 'Called node.replace_all_uses_with(replace_with, propagate_meta=True), but replace_with already has .meta keys'
        for k, v in self.meta.items():
            replace_with.meta[k] = v
    to_process = list(self.users)
    skipped = []
    for use_node in to_process:
        if not delete_user_cb(use_node):
            skipped.append(use_node)
            continue

        def maybe_replace_node(n: Node) -> Node:
            if n == self:
                return replace_with
            else:
                return n
        new_args = map_arg(use_node.args, maybe_replace_node)
        new_kwargs = map_arg(use_node.kwargs, maybe_replace_node)
        assert isinstance(new_args, tuple)
        assert isinstance(new_kwargs, dict)
        use_node.__update_args_kwargs(new_args, new_kwargs)
    assert len(self.users) - len(skipped) == 0
    return [n for n in to_process if n not in skipped]