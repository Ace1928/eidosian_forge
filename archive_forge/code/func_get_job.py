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
def get_job(id: str, address: Optional[str]=None, timeout: int=DEFAULT_RPC_TIMEOUT, _explain: bool=False) -> Optional[JobState]:
    """Get a submission job detail by id.

    Args:
        id: Submission ID obtained from job API.
        address: Ray bootstrap address, could be `auto`, `localhost:6379`.
            If None, it will be resolved automatically from an initialized ray.
        timeout: Max timeout value for the state API requests made.
        _explain: Print the API information such as API latency or
            failed query information.

    Returns:
        None if job not found, or
        :class:`JobState <ray.util.state.common.JobState>`.

    Raises:
        Exceptions: :class:`RayStateApiException <ray.util.state.exception.RayStateApiException>` if the CLI
            failed to query the data.
    """
    return StateApiClient(address=address).get(StateResource.JOBS, id, GetApiOptions(timeout=timeout), _explain=_explain)