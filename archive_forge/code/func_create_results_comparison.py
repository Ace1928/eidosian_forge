import torch
import torch.fx
from torch.fx import (
from torch.ao.ns.fx.utils import (
from torch.ao.ns.fx.ns_types import (
from torch.ao.ns.fx.graph_passes import _maybe_get_fqn
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization.utils import getattr_from_fqn
from torch.ao.quantization.fx.match_utils import _MatchResult
from torch.utils._pytree import tree_map
import collections
import copy
from typing import List, Dict, Set, Tuple, Callable, Any, Optional
import operator
def create_results_comparison(results_grouped) -> Any:
    """
    Input:

    {
      'subgraph_0': {
        '0': {
          'ref_node_name': '...',
          'ref_node_target_type': ...,
          'values': [torch.tensor(...), ...],
          'qconfig_str': '',
          'comparisons': [],
          'comparison_fn_name': '',
          'fqn': '...',
        },
        '1': {
          'ref_node_name': '...',
          'ref_node_target_type': ...,
          'values': [torch.tensor(...), ...],
          'qconfig_str': '...',
          'comparisons': [torch.tensor(...), ...],
          'comparison_fn_name': 'sqnr',
          'fqn': '...',
        },
      },
    }

    Output:
    {
      'subgraph_0': {
        'ref_node_name': '...',
        'ref_node_target_type': '...',
        'fqn': '...',
        'candidates': {
          '1': {
            'qconfig_str': ...,
            'comparison_fn_name': 'sqnr',
            'cmp_raw': [..., ...],
            'cmp_mean': ...,
          },
          ...,
        },
      },
    }
    """
    results_comparison = {}
    for subgraph_name, subgraph_results in results_grouped.items():
        candidates = {}
        for subgraph_inner_name, subgraph_inner_result in subgraph_results.items():
            if subgraph_inner_name == '0':
                continue
            cmp_raw = subgraph_inner_result['comparisons']
            cmp_raw_tensor = torch.stack(cmp_raw)
            candidates[subgraph_inner_name] = {'qconfig_str': subgraph_inner_result['qconfig_str'], 'comparison_fn_name': subgraph_inner_result['comparison_fn_name'], 'cmp_raw': cmp_raw_tensor, 'cmp_mean': torch.mean(cmp_raw_tensor)}
        results_comparison[subgraph_name] = {'ref_node_name': subgraph_results['0']['ref_node_name'], 'ref_node_target_type': subgraph_results['0']['ref_node_target_type'], 'fqn': subgraph_results['0']['fqn'], 'candidates': candidates}
    return results_comparison