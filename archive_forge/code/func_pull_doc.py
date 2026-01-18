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
def pull_doc(self, document: Document) -> None:
    """ Pull a document from the server, overwriting the passed-in document

        Args:
            document : (Document)
              The document to overwrite with server content.

        Returns:
            None

        """
    msg = self._protocol.create('PULL-DOC-REQ')
    reply = self._send_message_wait_for_reply(msg)
    if reply is None:
        raise RuntimeError('Connection to server was lost')
    elif reply.header['msgtype'] == 'ERROR':
        raise RuntimeError('Failed to pull document: ' + reply.content['text'])
    else:
        reply.push_to_document(document)