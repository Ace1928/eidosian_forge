from typing import TYPE_CHECKING, List
from twisted.trial.unittest import SynchronousTestCase
from .reactormixins import ReactorBuilder
def sleepThenStop() -> None:
    r.callFromThread(doStop)