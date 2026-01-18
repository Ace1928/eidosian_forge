import logging
import threading
import urllib
import warnings
from contextlib import contextmanager
from dataclasses import fields
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import requests
from ray.dashboard.modules.dashboard_sdk import SubmissionClient
from ray.dashboard.utils import (
from ray.util.annotations import DeveloperAPI
from ray.util.state.common import (
from ray.util.state.exception import RayStateApiException, ServerUnavailable
@DeveloperAPI
def list_cluster_events(address: Optional[str]=None, filters: Optional[List[Tuple[str, PredicateType, SupportedFilterType]]]=None, limit: int=DEFAULT_LIMIT, timeout: int=DEFAULT_RPC_TIMEOUT, detail: bool=False, raise_on_missing_output: bool=True, _explain: bool=False) -> List[Dict]:
    return StateApiClient(address=address).list(StateResource.CLUSTER_EVENTS, options=ListApiOptions(limit=limit, timeout=timeout, filters=filters, detail=detail), raise_on_missing_output=raise_on_missing_output, _explain=_explain)