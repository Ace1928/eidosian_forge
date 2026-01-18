from __future__ import annotations
import json
from typing import (
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run
def process_model(self, run: Run) -> Optional[Dict[str, Any]]:
    """Utility to process a run for wandb model_dict serialization.
        :param run: The run to process.
        :return: The convert model_dict to pass to WBTraceTree.
        """
    try:
        data = json.loads(run.json())
        processed = self.flatten_run(data)
        keep_keys = ('id', 'name', 'serialized', 'inputs', 'outputs', 'parent_run_id', 'execution_order')
        processed = self.truncate_run_iterative(processed, keep_keys=keep_keys)
        exact_keys, partial_keys = (('lc', 'type'), ('api_key',))
        processed = self.modify_serialized_iterative(processed, exact_keys=exact_keys, partial_keys=partial_keys)
        output = self.build_tree(processed)
        return output
    except Exception as e:
        if PRINT_WARNINGS:
            self.wandb.termwarn(f'WARNING: Failed to serialize model: {e}')
        return None