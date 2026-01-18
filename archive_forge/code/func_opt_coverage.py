import gc
import inspect
import os
import pdb
import random
import sys
import time
import trace
import warnings
from typing import NoReturn, Optional, Type
from twisted import plugin
from twisted.application import app
from twisted.internet import defer
from twisted.python import failure, reflect, usage
from twisted.python.filepath import FilePath
from twisted.python.reflect import namedModule
from twisted.trial import itrial, runner
from twisted.trial._dist.disttrial import DistTrialRunner
from twisted.trial.unittest import TestSuite
def opt_coverage(self):
    """
        Generate coverage information in the coverage file in the
        directory specified by the temp-directory option.
        """
    self.tracer = trace.Trace(count=1, trace=0)
    sys.settrace(self.tracer.globaltrace)
    self['coverage'] = True