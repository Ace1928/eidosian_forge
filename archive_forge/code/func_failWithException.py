import errno
import os
import sys
from typing import Optional
from twisted.trial.unittest import TestCase
def failWithException():
    raise ValueError('This does not work')