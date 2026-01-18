import ast
import io
import sys
import traceback
import testtools
from testtools.compat import (
from testtools.matchers import (
def test_custom_exception_no_args(self):
    """Reraising does not require args attribute to contain params"""

    class CustomException(Exception):
        """Exception that expects and sets attrs but not args"""

        def __init__(self, value):
            Exception.__init__(self)
            self.value = value
    try:
        raise CustomException('Some value')
    except CustomException:
        _exc_info = sys.exc_info()
    self.assertRaises(CustomException, reraise, *_exc_info)