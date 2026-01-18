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
def get_objects(id: str, address: Optional[str]=None, timeout: int=DEFAULT_RPC_TIMEOUT, _explain: bool=False) -> List[ObjectState]:
    """Get objects by id.

    There could be more than 1 entry returned since an object could be
    referenced at different places.

    Args:
        id: Id of the object
        address: Ray bootstrap address, could be `auto`, `localhost:6379`.
            If None, it will be resolved automatically from an initialized ray.
        timeout: Max timeout value for the state APIs requests made.
        _explain: Print the API information such as API latency or
            failed query information.

    Returns:
        List of
        :class:`~ray.util.state.common.ObjectState`.

    Raises:
        Exceptions: :class:`RayStateApiException <ray.util.state.exception.RayStateApiException>`  if the CLI
            failed to query the data.
    """
    return StateApiClient(address=address).get(StateResource.OBJECTS, id, GetApiOptions(timeout=timeout), _explain=_explain)