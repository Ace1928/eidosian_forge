import abc
from typing import Awaitable, Callable, Dict, Optional, Sequence, Union
from google.cloud.pubsublite_v1 import gapic_version as package_version
import google.auth  # type: ignore
import google.api_core
from google.api_core import exceptions as core_exceptions
from google.api_core import gapic_v1
from google.api_core import retry as retries
from google.auth import credentials as ga_credentials  # type: ignore
from google.oauth2 import service_account  # type: ignore
from google.cloud.pubsublite_v1.types import cursor
from google.longrunning import operations_pb2
@property
def commit_cursor(self) -> Callable[[cursor.CommitCursorRequest], Union[cursor.CommitCursorResponse, Awaitable[cursor.CommitCursorResponse]]]:
    raise NotImplementedError()