from random import random as _goodEnoughRandom
from typing import List
from automat import MethodicalMachine
from twisted.application import service
from twisted.internet import task
from twisted.internet.defer import (
from twisted.logger import Logger
from twisted.python import log
from twisted.python.failure import Failure
def handleIt(err):
    if self._raiseSynchronously:
        raisedNow.append(err)
    elif not err.check(CancelledError):
        log.err(err)