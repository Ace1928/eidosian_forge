import unittest
import re
from IPython.utils.capture import capture_output
import sys
import pytest
from tempfile import TemporaryDirectory
from IPython.testing import tools as tt
@pytest.mark.parametrize('outer_chain', ['none', 'from', 'another'])
@pytest.mark.parametrize('inner_chain', ['none', 'from', 'another'])
def test_native_exceptiongroup(outer_chain, inner_chain) -> None:
    pytest.importorskip('exceptiongroup')
    _exceptiongroup_common(outer_chain, inner_chain, native=False)