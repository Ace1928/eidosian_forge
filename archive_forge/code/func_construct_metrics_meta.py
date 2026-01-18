import time
import logging
import typing as tp
from IPython.display import display
from copy import deepcopy
from typing import List, Optional, Any, Union
from .ipythonwidget import MetricWidget
@staticmethod
def construct_metrics_meta(metrics: List[Union[str, tp.Dict[str, str]]]) -> List[tp.Dict[str, str]]:
    meta: List[tp.Dict[str, str]] = []
    for item in metrics:
        if isinstance(item, str):
            name, best_value = (item, 'Undefined')
        elif isinstance(item, dict):
            assert 'name' in item and 'best_value' in item, 'Wrong metrics definition format: should have `name` and `best_value` fields'
            name, best_value = (item['name'], item['best_value'])
        else:
            assert False, 'Each metric should be defined as str or asdict with `name` and `best_value` fields'
        meta.append({'best_value': best_value, 'name': name})
    return meta