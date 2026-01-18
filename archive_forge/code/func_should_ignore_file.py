from __future__ import annotations
import argparse
import ast
import functools
import logging
import tokenize
from typing import Any
from typing import Generator
from typing import List
from typing import Tuple
from flake8 import defaults
from flake8 import utils
from flake8._compat import FSTRING_END
from flake8._compat import FSTRING_MIDDLE
from flake8.plugins.finder import LoadedPlugin
def should_ignore_file(self) -> bool:
    """Check if ``flake8: noqa`` is in the file to be ignored.

        :returns:
            True if a line matches :attr:`defaults.NOQA_FILE`,
            otherwise False
        """
    if not self.options.disable_noqa and any((defaults.NOQA_FILE.match(line) for line in self.lines)):
        return True
    elif any((defaults.NOQA_FILE.search(line) for line in self.lines)):
        LOG.warning('Detected `flake8: noqa` on line with code. To ignore an error on a line use `noqa` instead.')
        return False
    else:
        return False