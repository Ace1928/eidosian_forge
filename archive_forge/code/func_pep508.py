import logging
import os
import re
import string
import typing
from itertools import chain as _chain
def pep508(value: str) -> bool:
    return True