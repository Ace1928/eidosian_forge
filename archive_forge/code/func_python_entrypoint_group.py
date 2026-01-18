import logging
import os
import re
import string
import typing
from itertools import chain as _chain
def python_entrypoint_group(value: str) -> bool:
    return ENTRYPOINT_GROUP_REGEX.match(value) is not None