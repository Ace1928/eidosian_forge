import bisect
import configparser
import inspect
import io
import keyword
import os
import re
import sys
import time
import tokenize
import warnings
from fnmatch import fnmatch
from functools import lru_cache
from optparse import OptionParser
@register_check
def bare_except(logical_line, noqa):
    """When catching exceptions, mention specific exceptions when
    possible.

    Okay: except Exception:
    Okay: except BaseException:
    E722: except:
    """
    if noqa:
        return
    match = BLANK_EXCEPT_REGEX.match(logical_line)
    if match:
        yield (match.start(), "E722 do not use bare 'except'")