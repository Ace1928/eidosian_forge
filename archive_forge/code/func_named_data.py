import codecs
import inspect
import json
import os
import re
from enum import Enum, unique
from functools import wraps
from collections.abc import Sequence
def named_data(*named_values):
    """
    This decorator is to allow for meaningful names to be given to tests that would otherwise use @ddt.data and
    @ddt.unpack.

    Example of original ddt usage:
        @ddt.ddt
        class TestExample(TemplateTest):
            @ddt.data(
                [0, 1],
                [10, 11]
            )
            @ddt.unpack
            def test_values(self, value1, value2):
                ...

    Example of new usage:
        @ddt.ddt
        class TestExample(TemplateTest):
            @named_data(
                ['LabelA', 0, 1],
                ['LabelB', 10, 11],
            )
            def test_values(self, value1, value2):
                ...

    Note that @unpack is not used.

    :param Sequence[Any] | dict[Any,Any] named_values: Each named_value should be a Sequence (e.g. list or tuple) with
        the name as the first element, or a dictionary with 'name' as one of the keys. The name will be coerced to a
        string and all other values will be passed unchanged to the test.
    """
    values = []
    for named_value in named_values:
        if not isinstance(named_value, (Sequence, dict)):
            raise TypeError("@named_data expects a Sequence (list, tuple) or dictionary, and not '{}'.".format(type(named_value)))
        value = _NamedDataDict(**named_value) if isinstance(named_value, dict) else _NamedDataList(named_value[0], *named_value[1:])
        value.__doc__ = None
        values.append(value)

    def wrapper(func):
        data(*values)(unpack(func))
        return func
    return wrapper