import ray
from ray.dag.base import DAGNodeBase
from ray.dag.py_obj_scanner import _PyObjScanner
from ray.util.annotations import DeveloperAPI
from typing import (
import uuid
import asyncio
def apply_functional(self, source_input_list: Any, predictate_fn: Callable, apply_fn: Callable):
    """
        Apply a given function to DAGNodes in source_input_list, and return
        the replaced inputs without mutating or coping any DAGNode.

        Args:
            source_input_list: Source inputs to extract and apply function on
                all children DAGNode instances.
            predictate_fn: Applied on each DAGNode instance found and determine
                if we should apply function to it. Can be used to filter node
                types.
            apply_fn: Function to appy on the node on bound attributes. Example:
                apply_fn = lambda node: node._get_serve_deployment_handle(
                    node._deployment, node._bound_other_args_to_resolve
                )

        Returns:
            replaced_inputs: Outputs of apply_fn on DAGNodes in
                source_input_list that passes predictate_fn.
        """
    replace_table = {}
    scanner = _PyObjScanner()
    for node in scanner.find_nodes(source_input_list):
        if predictate_fn(node) and node not in replace_table:
            replace_table[node] = apply_fn(node)
    replaced_inputs = scanner.replace_nodes(replace_table)
    scanner.clear()
    return replaced_inputs