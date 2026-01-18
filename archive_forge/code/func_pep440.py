import logging
import os
import re
import string
import typing
from itertools import chain as _chain
def pep440(version: str) -> bool:
    return VERSION_REGEX.match(version) is not None