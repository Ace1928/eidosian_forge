from __future__ import print_function
import functools
import os
import subprocess
from unittest import TestCase, skipIf
import attr
from .._methodical import MethodicalMachine
from .test_discover import isTwistedInstalled
def sampleMachine():
    """
    Create a sample L{MethodicalMachine} with some sample states.
    """
    mm = MethodicalMachine()

    class SampleObject(object):

        @mm.state(initial=True)
        def begin(self):
            """initial state"""

        @mm.state()
        def end(self):
            """end state"""

        @mm.input()
        def go(self):
            """sample input"""

        @mm.output()
        def out(self):
            """sample output"""
        begin.upon(go, end, [out])
    so = SampleObject()
    so.go()
    return mm