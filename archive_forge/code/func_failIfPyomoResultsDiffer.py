import sys
import os
import re
from inspect import getfile
import pyomo.common.unittest as unittest
import subprocess
def failIfPyomoResultsDiffer(self, cmd, baseline, cwd=None):
    _failIfPyomoResultsDiffer(self, cmd=cmd, baseline=baseline, cwd=cwd)