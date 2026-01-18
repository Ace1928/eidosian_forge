import logging
import os
import re
import string
import typing
from itertools import chain as _chain
def python_identifier(value: str) -> bool:
    return value.isidentifier()