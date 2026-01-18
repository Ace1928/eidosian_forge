from typing import Any, Callable, Dict, List, Tuple, Type, Union
from ._events import Data, EndOfMessage, Event, InformationalResponse, Request, Response
from ._headers import Headers
from ._state import CLIENT, IDLE, SEND_BODY, SEND_RESPONSE, SERVER
from ._util import LocalProtocolError, Sentinel
def send_eom(self, headers: Headers, write: Writer) -> None:
    if headers:
        raise LocalProtocolError("can't send trailers to HTTP/1.0 client")