from __future__ import annotations
import contextlib
import logging
import typing as t
import uuid
from traitlets.utils.importstring import import_item
import comm
def comm_close(self, stream: ZMQStream, ident: str, msg: MessageType) -> None:
    """Handler for comm_close messages"""
    content = msg['content']
    comm_id = content['comm_id']
    comm = self.get_comm(comm_id)
    if comm is None:
        return
    self.comms[comm_id]._closed = True
    del self.comms[comm_id]
    try:
        comm.handle_close(msg)
    except Exception:
        logger.error('Exception in comm_close for %s', comm_id, exc_info=True)