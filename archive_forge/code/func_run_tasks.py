import concurrent.futures as cf
import logging
import sys
from abc import ABC, abstractmethod
from enum import Enum
from threading import Event, RLock
from traceback import StackSummary, extract_stack
from typing import (
from uuid import uuid4
from adagio.exceptions import AbortedError, SkippedError, WorkflowBug
from adagio.specs import ConfigSpec, InputSpec, OutputSpec, TaskSpec, WorkflowSpec
from six import reraise  # type: ignore
from triad.collections.dict import IndexedOrderedDict, ParamDict
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_or_throw as aot
from triad.utils.convert import to_instance
from triad.utils.hash import to_uuid
def run_tasks(self, tasks: List['_Task']) -> None:
    if self._concurrency <= 1:
        for t in tasks:
            self.run_single(t)
        return
    with cf.ThreadPoolExecutor(max_workers=self._concurrency) as e:
        jobs = [e.submit(self.run_single, task) for task in tasks]
        while jobs:
            for f in cf.as_completed(jobs):
                jobs.remove(f)
                try:
                    for task in f.result().downstream:
                        jobs.append(e.submit(self.run_single, task))
                except Exception:
                    self.context.abort()
                    raise