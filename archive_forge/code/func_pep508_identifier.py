import logging
import os
import re
import string
import typing
from itertools import chain as _chain
def pep508_identifier(name: str) -> bool:
    return PEP508_IDENTIFIER_REGEX.match(name) is not None