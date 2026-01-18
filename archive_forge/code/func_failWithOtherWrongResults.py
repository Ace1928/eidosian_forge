import errno
import os
import sys
from typing import Optional
from twisted.trial.unittest import TestCase
def failWithOtherWrongResults():
    return [0, 1, 2]