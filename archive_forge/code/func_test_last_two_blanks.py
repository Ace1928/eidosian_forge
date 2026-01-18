import unittest
import pytest
import sys
from IPython.core.inputtransformer import InputTransformer
from IPython.core.tests.test_inputtransformer import syntax, syntax_ml
from IPython.testing import tools as tt
def test_last_two_blanks():
    assert isp.last_two_blanks('') is False
    assert isp.last_two_blanks('abc') is False
    assert isp.last_two_blanks('abc\n') is False
    assert isp.last_two_blanks('abc\n\na') is False
    assert isp.last_two_blanks('abc\n \n') is False
    assert isp.last_two_blanks('abc\n\n') is False
    assert isp.last_two_blanks('\n\n') is True
    assert isp.last_two_blanks('\n\n ') is True
    assert isp.last_two_blanks('\n \n') is True
    assert isp.last_two_blanks('abc\n\n ') is True
    assert isp.last_two_blanks('abc\n\n\n') is True
    assert isp.last_two_blanks('abc\n\n \n') is True
    assert isp.last_two_blanks('abc\n\n \n ') is True
    assert isp.last_two_blanks('abc\n\n \n \n') is True
    assert isp.last_two_blanks('abc\nd\n\n\n') is True
    assert isp.last_two_blanks('abc\nd\ne\nf\n\n\n') is True