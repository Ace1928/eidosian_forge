from __future__ import annotations
from collections import deque
from typing import Callable, Optional, Set
from zope.interface import implementer
from . import IWorker
from ._convenience import Quit
from ._ithreads import IExclusiveWorker
@self._coordinator.do
def createOneWorker() -> None:
    for x in range(n):
        worker = self._createWorker()
        if worker is None:
            return
        self._recycleWorker(worker)