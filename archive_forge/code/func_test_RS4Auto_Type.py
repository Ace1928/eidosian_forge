import pytest
import sys
import textwrap
import rpy2.robjects as robjects
import rpy2.robjects.methods as methods
def test_RS4Auto_Type():
    robjects.r('library(stats4)')

    class MLE(object, metaclass=robjects.methods.RS4Auto_Type):
        __rname__ = 'mle'
        __rpackagename__ = 'stats4'