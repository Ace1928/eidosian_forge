import logging
import threading
import uuid
from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, Optional
from ..lib import mailbox, tracelog
from .message_future import MessageFuture
def send_and_receive(self, rec: 'pb.Record', local: Optional[bool]=None) -> MessageFuture:
    rec.control.req_resp = True
    if local:
        rec.control.local = local
    rec.uuid = uuid.uuid4().hex
    future = MessageFutureObject()
    with self._lock:
        self._pending_reqs[rec.uuid] = future
    self._send_message(rec)
    return future