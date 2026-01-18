import collections
import logging
import operator
from typing import Any, DefaultDict, Deque, Dict, Iterator, List, Optional, Set, Tuple
import torch
from torch._dynamo.utils import counters
from torch._utils_internal import print_graph
from .. import config
from ..pattern_matcher import (
def apply_group_batch_fusion(graph: torch.fx.GraphModule, rule: GroupBatchFusionBase):
    stable_topological_sort(graph)
    fused_set: Set[torch.fx.Node] = set()
    for node in reversed(graph.nodes):
        candidates = get_fusion_candidates(rule, node, fused_set)
        for key, candidate_nodes in candidates.items():
            if len(candidate_nodes) < MIN_FUSE_SET_SIZE:
                continue
            for subset in find_independent_subset_greedy(candidate_nodes, rule.graph_search_options):
                rule.fuse(graph, subset)
                fused_set.update(subset)
                if isinstance(rule, GroupFusion):
                    counters['inductor']['group_fusion'] += 1
                elif isinstance(rule, BatchFusion):
                    counters['inductor']['batch_fusion'] += 1
                else:
                    counters['inductor']['unknown_group_batch_fusion'] += 1
                log.info(f'{rule.__class__.__name__}: key = {key}; subset size = {len(subset)}')