import gc
import sys
from asyncio import (
from unittest import skipIf
from twisted.internet.asyncioreactor import AsyncioSelectorReactor
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase
from .reactormixins import ReactorBuilder
def newLoop(self, policy: AbstractEventLoopPolicy) -> AbstractEventLoop:
    """
        Make a new asyncio loop from a policy for use with a reactor, and add
        appropriate cleanup to restore any global state.
        """
    existingLoop = get_event_loop()
    existingPolicy = get_event_loop_policy()
    result = policy.new_event_loop()

    @self.addCleanup
    def cleanUp():
        result.close()
        set_event_loop(existingLoop)
        set_event_loop_policy(existingPolicy)
    return result