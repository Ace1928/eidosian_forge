from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any, Callable
from tornado.httpclient import HTTPClientError, HTTPRequest
from tornado.ioloop import IOLoop
from tornado.websocket import WebSocketError, websocket_connect
from ..core.types import ID
from ..protocol import Protocol
from ..protocol.exceptions import MessageError, ProtocolError, ValidationError
from ..protocol.receiver import Receiver
from ..util.strings import format_url_query_arguments
from ..util.tornado import fixup_windows_event_loop_policy
from .states import (
from .websocket import WebSocketClientConnectionWrapper
def push_doc(self, document: Document) -> Message[Any]:
    """ Push a document to the server, overwriting any existing server-side doc.

        Args:
            document : (Document)
                A Document to push to the server

        Returns:
            The server reply

        """
    msg = self._protocol.create('PUSH-DOC', document)
    reply = self._send_message_wait_for_reply(msg)
    if reply is None:
        raise RuntimeError('Connection to server was lost')
    elif reply.header['msgtype'] == 'ERROR':
        raise RuntimeError('Failed to push document: ' + reply.content['text'])
    else:
        return reply