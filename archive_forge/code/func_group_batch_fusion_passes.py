import collections
import logging
import operator
from typing import Any, DefaultDict, Deque, Dict, Iterator, List, Optional, Set, Tuple
import torch
from torch._dynamo.utils import counters
from torch._utils_internal import print_graph
from .. import config
from ..pattern_matcher import (
def group_batch_fusion_passes(graph: torch.fx.Graph, pre_grad=True):
    print_graph(graph, 'Before group_batch fusion in pre grad pass.')
    fusions: List[GroupBatchFusionBase] = []
    if pre_grad:
        fusions += generate_fusion_from_config(config.pre_grad_fusion_options, pre_grad=True)
    else:
        fbgemm_fusion_keys = [x for x in config.post_grad_fusion_options if config.post_grad_fusion_options[x].get('require_fbgemm', False)]
        fbgemm_fusions = {fusion: config.post_grad_fusion_options[fusion] for fusion in fbgemm_fusion_keys}
        non_fbgemm_fusions = {fusion: config.post_grad_fusion_options[fusion] for fusion in config.post_grad_fusion_options.keys() if fusion not in fbgemm_fusion_keys}
        fusions += generate_fusion_from_config(non_fbgemm_fusions, pre_grad=False)
        if has_fbgemm:
            fusions += generate_fusion_from_config(fbgemm_fusions, pre_grad=False)
    for rule in fusions:
        apply_group_batch_fusion(graph, rule)
        print_graph(graph, f'Apply fusion {rule.__class__.__name__}.')