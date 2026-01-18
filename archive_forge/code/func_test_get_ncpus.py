import multiprocessing
import unittest
import warnings
import pytest
from monty.dev import deprecated, get_ncpus, install_excepthook, requires
def test_get_ncpus(self):
    assert get_ncpus() == multiprocessing.cpu_count()