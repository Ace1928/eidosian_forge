import torch._dynamo.test_case
import unittest.mock
import os
import contextlib
import torch._logging
import torch._logging._internal
from torch._dynamo.utils import LazyString
import logging
def kwargs_to_settings(**kwargs):
    INT_TO_VERBOSITY = {10: '+', 20: '', 40: '-'}
    settings = []

    def append_setting(name, level):
        if isinstance(name, str) and isinstance(level, int) and (level in INT_TO_VERBOSITY):
            settings.append(INT_TO_VERBOSITY[level] + name)
            return
        else:
            raise ValueError('Invalid value for setting')
    for name, val in kwargs.items():
        if isinstance(val, bool):
            settings.append(name)
        elif isinstance(val, int):
            append_setting(name, val)
        elif isinstance(val, dict) and name == 'modules':
            for module_qname, level in val.items():
                append_setting(module_qname, level)
        else:
            raise ValueError('Invalid value for setting')
    return ','.join(settings)