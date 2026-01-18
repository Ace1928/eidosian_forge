from __future__ import annotations
from itertools import count
from typing import TYPE_CHECKING
from . import messaging
from .entity import Exchange, Queue
def iterqueue(self, limit=None, infinite=False):
    for items_since_start in count():
        item = self.fetch()
        if not infinite and item is None or (limit and items_since_start >= limit):
            break
        yield item