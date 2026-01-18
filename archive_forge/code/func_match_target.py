import os
import re
import string
import textwrap
import warnings
from string import Formatter
from pathlib import Path
from typing import List, Dict, Tuple, Optional, cast
def match_target(s):
    if field is None:
        return s
    parts = s.split()
    try:
        tgt = parts[field]
        return tgt
    except IndexError:
        return ''