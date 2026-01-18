from pathlib import Path
import pytest
import textwrap
from . import util
from numpy.f2py import crackfortran
from numpy.testing import IS_WASM
def test_abstract_interface(self):
    assert self.module.ops_module.foo(3, 5) == (8, 13)