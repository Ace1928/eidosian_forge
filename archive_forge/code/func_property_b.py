import multiprocessing
import unittest
import warnings
import pytest
from monty.dev import deprecated, get_ncpus, install_excepthook, requires
@property
@deprecated(property_a)
def property_b(self):
    return 'b'