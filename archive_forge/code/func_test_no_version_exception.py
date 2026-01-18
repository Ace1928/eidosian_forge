import sys
import pyomo.common.unittest as unittest
from pyomo.common import DeveloperError
from pyomo.common.deprecation import (
from pyomo.common.log import LoggingIntercept
from io import StringIO
import logging
def test_no_version_exception(self):
    with self.assertRaisesRegex(DeveloperError, "@deprecated\\(\\): missing 'version' argument"):

        @deprecated()
        def foo():
            pass
    with self.assertRaisesRegex(DeveloperError, "@deprecated\\(\\): missing 'version' argument"):

        @deprecated()
        class foo(object):
            pass

    @deprecated()
    class foo(object):

        @deprecated(version='1.2')
        def __init__(self):
            pass
    self.assertIn('.. deprecated:: 1.2', foo.__doc__)