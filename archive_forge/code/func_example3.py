from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import core
from fire import decorators
from fire import testutils
@decorators.SetParseFns(float, arg2=str)
def example3(self, arg1, arg2):
    return (arg1, arg2)