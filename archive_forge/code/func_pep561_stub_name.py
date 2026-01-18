import logging
import os
import re
import string
import typing
from itertools import chain as _chain
def pep561_stub_name(value: str) -> bool:
    top, *children = value.split('.')
    if not top.endswith('-stubs'):
        return False
    return python_module_name('.'.join([top[:-len('-stubs')], *children]))