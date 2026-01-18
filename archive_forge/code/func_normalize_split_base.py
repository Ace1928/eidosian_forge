import itertools
import logging
import operator
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union
from typing_extensions import TypeAlias
import torch
from torch._dynamo.utils import counters
from ..pattern_matcher import (
from .pre_grad import (
def normalize_split_base(match: Match, _get_split_args: Callable[[torch.fx.Node], Tuple[Optional[torch.fx.Node], Optional[Any], Optional[int]]]):
    """
    Normalize split with split_size into split_with_sizes, so that we only deal with one type of split in
    subsequent optimizations
    """
    split_node = match.nodes[0]
    graph = match.graph
    split_input, split_size, split_dim = _get_split_args(split_node)
    if split_input is None or split_dim is None or split_size is None:
        log.info("couldn't find split args")
        return
    if 'example_value' not in split_node.meta:
        log.warning('example value absent for node: %s', split_node)
        return
    assert isinstance(split_node.meta['example_value'], (list, tuple))
    split_sections = [t.size()[split_dim] for t in split_node.meta['example_value']]
    if any((isinstance(section, torch.SymInt) for section in split_sections)):
        return
    if len(split_sections) == 1:
        remove_split_with_size_one(graph, split_node, split_input)
        return
    if split_dim < 0:
        split_dim += split_input.meta['example_value'].dim()
    with graph.inserting_after(split_node):
        new_split_node = graph.call_function(torch.split, args=(split_input, split_sections), kwargs={'dim': split_dim})
    split_node.replace_all_uses_with(new_split_node)
    new_split_node.meta.update(split_node.meta)
    graph.erase_node(split_node)
    counters['inductor']['split_cat_norm'] += 1