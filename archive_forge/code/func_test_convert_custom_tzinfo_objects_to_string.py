from collections import OrderedDict
from collections.abc import Iterator
from functools import partial
import datetime
import sys
import pytest
import hypothesis as h
import hypothesis.strategies as st
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.types as types
import pyarrow.tests.strategies as past
def test_convert_custom_tzinfo_objects_to_string():

    class CorrectTimezone1(datetime.tzinfo):
        """
        Conversion is using utcoffset()
        """

        def tzname(self, dt):
            return None

        def utcoffset(self, dt):
            return datetime.timedelta(hours=-3, minutes=30)

    class CorrectTimezone2(datetime.tzinfo):
        """
        Conversion is using tzname()
        """

        def tzname(self, dt):
            return '+03:00'

        def utcoffset(self, dt):
            return datetime.timedelta(hours=3)

    class BuggyTimezone1(datetime.tzinfo):
        """
        Unable to infer name or offset
        """

        def tzname(self, dt):
            return None

        def utcoffset(self, dt):
            return None

    class BuggyTimezone2(datetime.tzinfo):
        """
        Wrong offset type
        """

        def tzname(self, dt):
            return None

        def utcoffset(self, dt):
            return 'one hour'

    class BuggyTimezone3(datetime.tzinfo):
        """
        Wrong timezone name type
        """

        def tzname(self, dt):
            return 240

        def utcoffset(self, dt):
            return None
    assert pa.lib.tzinfo_to_string(CorrectTimezone1()) == '-02:30'
    assert pa.lib.tzinfo_to_string(CorrectTimezone2()) == '+03:00'
    msg = 'Object returned by tzinfo.utcoffset\\(None\\) is not an instance of datetime.timedelta'
    for wrong in [BuggyTimezone1(), BuggyTimezone2(), BuggyTimezone3()]:
        with pytest.raises(ValueError, match=msg):
            pa.lib.tzinfo_to_string(wrong)