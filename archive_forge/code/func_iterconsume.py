from __future__ import annotations
from itertools import count
from typing import TYPE_CHECKING
from . import messaging
from .entity import Exchange, Queue
def iterconsume(self, limit=None, no_ack=False):
    return _iterconsume(self.connection, self, no_ack, limit)