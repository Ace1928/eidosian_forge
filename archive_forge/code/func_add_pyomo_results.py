import sys
import os
import re
from inspect import getfile
import pyomo.common.unittest as unittest
import subprocess
def add_pyomo_results(cls, name=None, cmd=None, fn=None, baseline=None, cwd=None):
    if cmd is None and fn is None:
        print("ERROR: must specify either the 'cmd' or 'fn' option to define how the output file is generated")
        return
    if name is None and baseline is None:
        print('ERROR: must specify a baseline comparison file, or the test name')
        return
    if baseline is None:
        baseline = name + '.txt'
    tmp = name.replace('/', '_')
    tmp = tmp.replace('\\', '_')
    tmp = tmp.replace('.', '_')
    if fn is None:
        func = lambda self, c1=cwd, c2=cmd, c3=tmp + '.out', c4=baseline: _failIfPyomoResultsDiffer(self, cwd=c1, cmd=c2, baseline=c4)
    else:
        sys.exit(1)
        func = lambda self, c1=fn, c2=tmp, c3=baseline: _failIfPyomoResultsDiffer(self, fn=c1, name=c2, baseline=c3)
    func.__name__ = 'test_' + tmp
    func.__doc__ = 'pyomo result test: ' + func.__name__ + ' (' + str(cls.__module__) + '.' + str(cls.__name__) + ')'
    setattr(cls, 'test_' + tmp, func)