import multiprocessing
import unittest
import warnings
import pytest
from monty.dev import deprecated, get_ncpus, install_excepthook, requires
@classmethod
@deprecated(classmethod_a, category=DeprecationWarning)
def classmethod_b(self):
    return 'b'