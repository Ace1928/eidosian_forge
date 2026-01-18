import asyncio
from collections import deque, defaultdict
import dataclasses
from dataclasses import field
import logging
from typing import List, Dict, Optional, Set, Deque, Callable
import ray
from ray.workflow.common import (
from ray.workflow.workflow_context import WorkflowTaskContext
def append_frontier_to_run(self, task_id: TaskID) -> None:
    """Insert one task to the frontier queue."""
    if task_id not in self.frontier_to_run_set and task_id not in self.running_frontier_set:
        self.frontier_to_run.append(task_id)
        self.frontier_to_run_set.add(task_id)