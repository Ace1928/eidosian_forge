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
def group_results_by_subgraph(results: NSResultsType) -> Any:
    """
    Creates a comparison of results

    Input:

    {
      'model': {
        'node_output': {
          'subgraph_0_0': [
            'values': [torch.tensor(...), ...], ...
            'ref_node_name': ...,
            'ref_node_target_type': ...,
            'qconfig_str': ...,
            'comparisons': [], ...
            'comparison_fn_name': '',
            'fqn': '...',
          ],
          'subgraph_0_1': [
            'values': [torch.tensor(...), ...], ...
            'ref_node_name': ...,
            'ref_node_target_type': ...,
            'qconfig_str': ...,
            'comparisons': [torch.tensor(...), ...], ...
            'comparison_fn_name': '...',
            'fqn': '...',
          ],
          ...
        },
      },
    }

    Output:
    {
      'subgraph_0': {
        '0': {
          'ref_node_name': '...',
          'ref_node_target_type': ...,
          'values': [torch.tensor(...), ...],
          'qconfig_str': None,
          'comparisons': [torch.tensor(...), ...], ...
          'comparison_fn_name': '...',
          'fqn': '...',
        },
        '1': {
          'ref_node_name': '...',
          'ref_node_target_type': ...,
          'values': [torch.tensor(...), ...],
          'qconfig_str': '...',
          'comparisons': [torch.tensor(...), ...], ...
          'comparison_fn_name': '...',
          'fqn': '...',
        },
      },
    }

    """
    subgraph_name_to_subgraph_results: Any = collections.defaultdict(dict)
    key_to_use = next(iter(results['model'].keys()))
    for subgraph_name_with_idx, subgraph_candidate_results in results['model'][key_to_use].items():
        subgraph_str, subgraph_idx, subgraph_candidate_idx = subgraph_name_with_idx.split('_')
        subgraph_name = f'{subgraph_str}_{subgraph_idx}'
        subgraph_results = {'ref_node_name': subgraph_candidate_results[0]['ref_node_name'], 'ref_node_target_type': subgraph_candidate_results[0]['ref_node_target_type'], 'fqn': subgraph_candidate_results[0]['fqn'], 'values': subgraph_candidate_results[0]['values'], 'qconfig_str': subgraph_candidate_results[0]['qconfig_str'], 'comparisons': subgraph_candidate_results[0]['comparisons'], 'comparison_fn_name': subgraph_candidate_results[0]['comparison_fn_name']}
        subgraph_name_to_subgraph_results[subgraph_name][subgraph_candidate_idx] = subgraph_results
    return dict(subgraph_name_to_subgraph_results)