from __future__ import annotations
import asyncio
import datetime
import logging
import pathlib
import uuid
from typing import (
import langsmith
from langsmith import run_helpers as rh
from langsmith import run_trees, schemas
from langsmith import utils as ls_utils
from langsmith._internal import _aiter as aitertools
from langsmith.beta import warn_beta
from langsmith.evaluation._runner import (
from langsmith.evaluation.evaluator import EvaluationResults, RunEvaluator
def aget_examples(self) -> AsyncIterator[schemas.Example]:
    if self._examples is None:
        self._examples = _aresolve_data(self._data, client=self.client)
    self._examples, examples_iter = aitertools.atee(aitertools.ensure_async_iterator(self._examples), 2, lock=asyncio.Lock())
    return examples_iter