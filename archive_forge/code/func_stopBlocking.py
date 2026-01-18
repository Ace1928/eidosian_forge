from typing import TYPE_CHECKING, List
from twisted.trial.unittest import SynchronousTestCase
from .reactormixins import ReactorBuilder
def stopBlocking() -> None:
    r2.callLater(0, r2stop)