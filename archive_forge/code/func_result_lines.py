import contextlib
import pathlib
from pathlib import Path
import re
import time
from typing import Union
from unittest import mock
def result_lines(result):
    return [x.strip() for x in re.split('\\r?\\n', re.sub(' +', ' ', result)) if x.strip() != '']