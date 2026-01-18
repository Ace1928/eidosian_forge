import pytest
import sys
import textwrap
import rpy2.robjects as robjects
import rpy2.robjects.methods as methods
def test_getclassdef():
    robjects.r('library(stats4)')
    cr = methods.getclassdef('mle', packagename='stats4')
    assert not cr.virtual