import logging
import os
import re
import string
import typing
from itertools import chain as _chain
def python_qualified_identifier(value: str) -> bool:
    if value.startswith('.') or value.endswith('.'):
        return False
    return all((python_identifier(m) for m in value.split('.')))