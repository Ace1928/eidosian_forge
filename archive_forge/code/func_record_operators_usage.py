import json
import re
import threading
from typing import Dict
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data._internal.logical.operators.map_operator import AbstractUDFMap
from ray.data._internal.logical.operators.read_operator import Read
from ray.data._internal.logical.operators.write_operator import Write
def record_operators_usage(op: LogicalOperator):
    """Record logical operator usage with Ray telemetry."""
    ops_dict = dict()
    _collect_operators_to_dict(op, ops_dict)
    ops_json_str = ''
    with _recorded_operators_lock:
        for op, count in ops_dict.items():
            _recorded_operators.setdefault(op, 0)
            _recorded_operators[op] += count
        ops_json_str = json.dumps(_recorded_operators)
    record_extra_usage_tag(TagKey.DATA_LOGICAL_OPS, ops_json_str)