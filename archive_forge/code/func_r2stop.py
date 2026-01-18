from typing import TYPE_CHECKING, List
from twisted.trial.unittest import SynchronousTestCase
from .reactormixins import ReactorBuilder
def r2stop() -> None:
    r2.stop()