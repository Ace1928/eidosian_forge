import abc
import ctypes
import inspect
import json
import warnings
from collections import OrderedDict
from copy import deepcopy
from enum import Enum
from functools import wraps
from os import SEEK_END, environ
from os.path import getsize
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
import scipy.sparse
from .compat import (PANDAS_INSTALLED, PYARROW_INSTALLED, arrow_cffi, arrow_is_floating, arrow_is_integer, concat,
from .libpath import find_lib_path
def create_node_record(tree: Dict[str, Any], node_depth: int=1, tree_index: Optional[int]=None, feature_names: Optional[List[str]]=None, parent_node: Optional[str]=None) -> Dict[str, Any]:

    def _get_node_index(tree: Dict[str, Any], tree_index: Optional[int]) -> str:
        tree_num = f'{tree_index}-' if tree_index is not None else ''
        is_split = _is_split_node(tree)
        node_type = 'S' if is_split else 'L'
        node_num = tree.get('split_index' if is_split else 'leaf_index', 0)
        return f'{tree_num}{node_type}{node_num}'

    def _get_split_feature(tree: Dict[str, Any], feature_names: Optional[List[str]]) -> Optional[str]:
        if _is_split_node(tree):
            if feature_names is not None:
                feature_name = feature_names[tree['split_feature']]
            else:
                feature_name = tree['split_feature']
        else:
            feature_name = None
        return feature_name

    def _is_single_node_tree(tree: Dict[str, Any]) -> bool:
        return set(tree.keys()) == {'leaf_value'}
    node: Dict[str, Union[int, str, None]] = OrderedDict()
    node['tree_index'] = tree_index
    node['node_depth'] = node_depth
    node['node_index'] = _get_node_index(tree, tree_index)
    node['left_child'] = None
    node['right_child'] = None
    node['parent_index'] = parent_node
    node['split_feature'] = _get_split_feature(tree, feature_names)
    node['split_gain'] = None
    node['threshold'] = None
    node['decision_type'] = None
    node['missing_direction'] = None
    node['missing_type'] = None
    node['value'] = None
    node['weight'] = None
    node['count'] = None
    if _is_split_node(tree):
        node['left_child'] = _get_node_index(tree['left_child'], tree_index)
        node['right_child'] = _get_node_index(tree['right_child'], tree_index)
        node['split_gain'] = tree['split_gain']
        node['threshold'] = tree['threshold']
        node['decision_type'] = tree['decision_type']
        node['missing_direction'] = 'left' if tree['default_left'] else 'right'
        node['missing_type'] = tree['missing_type']
        node['value'] = tree['internal_value']
        node['weight'] = tree['internal_weight']
        node['count'] = tree['internal_count']
    else:
        node['value'] = tree['leaf_value']
        if not _is_single_node_tree(tree):
            node['weight'] = tree['leaf_weight']
            node['count'] = tree['leaf_count']
    return node