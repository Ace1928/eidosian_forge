from __future__ import annotations
from itertools import count
from typing import TYPE_CHECKING
from . import messaging
from .entity import Exchange, Queue
def process_next(self):
    raise NotImplementedError('Use fetch(enable_callbacks=True)')