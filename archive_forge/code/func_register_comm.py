from __future__ import annotations
import contextlib
import logging
import typing as t
import uuid
from traitlets.utils.importstring import import_item
import comm
def register_comm(self, comm: BaseComm) -> str:
    """Register a new comm"""
    comm_id = comm.comm_id
    self.comms[comm_id] = comm
    return comm_id