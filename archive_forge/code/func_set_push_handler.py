from logging import getLogger
from typing import Any, Union
from ..exceptions import ConnectionError, InvalidResponse, ResponseError
from ..typing import EncodableT
from .base import _AsyncRESPBase, _RESPBase
from .socket import SERVER_CLOSED_CONNECTION_ERROR
def set_push_handler(self, push_handler_func):
    self.push_handler_func = push_handler_func