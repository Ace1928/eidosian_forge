import datetime
import operator
import warnings
import pytest
import tempfile
import re
import sys
import numpy as np
from numpy.testing import (
from numpy.core._multiarray_tests import fromstring_null_term_c_api
def assert_not_deprecated(self, function, args=(), kwargs={}):
    """Test that warnings are not raised.

        This is just a shorthand for:

        self.assert_deprecated(function, num=0, ignore_others=True,
                        exceptions=tuple(), args=args, kwargs=kwargs)
        """
    self.assert_deprecated(function, num=0, ignore_others=True, exceptions=tuple(), args=args, kwargs=kwargs)