import sys
from abc import ABC
from asyncio import IncompleteReadError, StreamReader, TimeoutError
from typing import List, Optional, Union
from ..exceptions import (
from ..typing import EncodableT
from .encoders import Encoder
from .socket import SERVER_CLOSED_CONNECTION_ERROR, SocketBuffer
@classmethod
def parse_error(cls, response):
    """Parse an error response"""
    error_code = response.split(' ')[0]
    if error_code in cls.EXCEPTION_CLASSES:
        response = response[len(error_code) + 1:]
        exception_class = cls.EXCEPTION_CLASSES[error_code]
        if isinstance(exception_class, dict):
            exception_class = exception_class.get(response, ResponseError)
        return exception_class(response)
    return ResponseError(response)