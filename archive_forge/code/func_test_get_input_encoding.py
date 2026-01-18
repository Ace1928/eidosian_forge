import unittest
import pytest
import sys
from IPython.core.inputtransformer import InputTransformer
from IPython.core.tests.test_inputtransformer import syntax, syntax_ml
from IPython.testing import tools as tt
def test_get_input_encoding():
    encoding = isp.get_input_encoding()
    assert isinstance(encoding, str)
    assert 'test'.encode(encoding) == b'test'