from __future__ import annotations
import collections
import concurrent.futures as cf
import datetime
import functools
import itertools
import logging
import pathlib
import threading
import uuid
from contextvars import copy_context
from typing import (
from requests import HTTPError
from typing_extensions import TypedDict
import langsmith
from langsmith import env as ls_env
from langsmith import run_helpers as rh
from langsmith import run_trees, schemas
from langsmith import utils as ls_utils
from langsmith.evaluation.evaluator import (
from langsmith.evaluation.integrations import LangChainStringEvaluator
def get_summary_scores(self) -> Dict[str, List[dict]]:
    """If summary_evaluators were applied, consume and return the results."""
    if self._summary_results is None:
        return {'results': []}
    return {'results': [res for results in self._summary_results for res in results['results']]}