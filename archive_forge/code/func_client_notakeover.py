import asyncio
import sys
from typing import Any, Optional, cast
from .client_exceptions import ClientError
from .client_reqrep import ClientResponse
from .helpers import call_later, set_result
from .http import (
from .http_websocket import WebSocketWriter  # WSMessage
from .streams import EofStream, FlowControlDataQueue
from .typedefs import (
@property
def client_notakeover(self) -> bool:
    return self._client_notakeover