import logging
import os
import re
import string
import typing
from itertools import chain as _chain
def pep517_backend_reference(value: str) -> bool:
    module, _, obj = value.partition(':')
    identifiers = (i.strip() for i in _chain(module.split('.'), obj.split('.')))
    return all((python_identifier(i) for i in identifiers if i))