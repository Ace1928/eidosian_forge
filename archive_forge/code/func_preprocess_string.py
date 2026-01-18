import collections
import contextlib
import doctest
import functools
import importlib
import inspect
import logging
import multiprocessing
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from collections import defaultdict
from collections.abc import Mapping
from io import StringIO
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Union
from unittest import mock
from unittest.mock import patch
import urllib3
from transformers import logging as transformers_logging
from .integrations import (
from .integrations.deepspeed import is_deepspeed_available
from .utils import (
import asyncio  # noqa
def preprocess_string(string, skip_cuda_tests):
    """Prepare a docstring or a `.md` file to be run by doctest.

    The argument `string` would be the whole file content if it is a `.md` file. For a python file, it would be one of
    its docstring. In each case, it may contain multiple python code examples. If `skip_cuda_tests` is `True` and a
    cuda stuff is detective (with a heuristic), this method will return an empty string so no doctest will be run for
    `string`.
    """
    codeblock_pattern = '(```(?:python|py)\\s*\\n\\s*>>> )((?:.*?\\n)*?.*?```)'
    codeblocks = re.split(re.compile(codeblock_pattern, flags=re.MULTILINE | re.DOTALL), string)
    is_cuda_found = False
    for i, codeblock in enumerate(codeblocks):
        if 'load_dataset(' in codeblock and '# doctest: +IGNORE_RESULT' not in codeblock:
            codeblocks[i] = re.sub('(>>> .*load_dataset\\(.*)', '\\1 # doctest: +IGNORE_RESULT', codeblock)
        if ('>>>' in codeblock or '...' in codeblock) and re.search('cuda|to\\(0\\)|device=0', codeblock) and skip_cuda_tests:
            is_cuda_found = True
            break
    modified_string = ''
    if not is_cuda_found:
        modified_string = ''.join(codeblocks)
    return modified_string