import multiprocessing
import unittest
import warnings
import pytest
from monty.dev import deprecated, get_ncpus, install_excepthook, requires
def test_deprecated_property(self):

    class a:

        def __init__(self):
            pass

        @property
        def property_a(self):
            pass

        @property
        @deprecated(property_a)
        def property_b(self):
            return 'b'

        @deprecated(property_a)
        def func_a(self):
            return 'a'
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        assert a().property_b == 'b'
        assert issubclass(w[-1].category, FutureWarning)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        assert a().func_a() == 'a'
        assert issubclass(w[-1].category, FutureWarning)