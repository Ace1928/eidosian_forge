import itertools
import logging
import operator
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union
from typing_extensions import TypeAlias
import torch
from torch._dynamo.utils import counters
from ..pattern_matcher import (
from .pre_grad import (
def get_transform_params(self, unbind_node: torch.fx.Node, next_users: List[torch.fx.Node], user_inputs_list: List[List[Union[torch.fx.Node, _Range]]]) -> Optional[List[List[_TransformParam]]]:
    """
        Figure out what transforms are needed for each input to each cat node.

        Here is the rough transforms we apply:

        x -> unbind -> stack => x -> movedim

        x -> unbind -> cat => x -> movedim -> flatten

        When cat/stack nodes have additional args:

             addn ---|              addn -> unsqueeze ---|
        x -> unbind -> stack  =>           x -> movedim  -> cat

             addn ---|                            addn ---|
        x -> unbind -> cat  =>   x -> movedim -> flatten  -> cat

        (Note application of these depends on the dims as well)


        """
    split_dim = unbind_node.kwargs['dim']
    transform_params_list: List[List[_TransformParam]] = []
    for user_node, user_inputs in zip(next_users, user_inputs_list):
        cat_dim = get_arg_value(user_node, 1, 'dim') or 0
        transform_params: List[_TransformParam] = []
        for user_input in user_inputs:
            if isinstance(user_input, tuple):
                movedim_params = (split_dim, cat_dim) if split_dim != cat_dim else None
                flatten_params = None
                if user_node.target == torch.cat:
                    flatten_params = (cat_dim, cat_dim + 1)
                transform_params.append((None, movedim_params, None, flatten_params))
            elif user_node.target == torch.stack:
                transform_params.append((None, None, (cat_dim,), None))
            else:
                transform_params.append((None, None, None, None))
        transform_params_list.append(transform_params)
    return transform_params_list