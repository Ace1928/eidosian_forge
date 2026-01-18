from logging import getLogger
from typing import Any, Union
from ..exceptions import ConnectionError, InvalidResponse, ResponseError
from ..typing import EncodableT
from .base import _AsyncRESPBase, _RESPBase
from .socket import SERVER_CLOSED_CONNECTION_ERROR
def handle_push_response(self, response):
    logger = getLogger('push_response')
    logger.info('Push response: ' + str(response))
    return response