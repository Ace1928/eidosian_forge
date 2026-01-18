import logging
import optparse
import shutil
import sys
import textwrap
from contextlib import suppress
from typing import Any, Dict, Generator, List, Tuple
from pip._internal.cli.status_codes import UNKNOWN_ERROR
from pip._internal.configuration import Configuration, ConfigurationError
from pip._internal.utils.misc import redact_auth_from_url, strtobool
def insert_option_group(self, idx: int, *args: Any, **kwargs: Any) -> optparse.OptionGroup:
    """Insert an OptionGroup at a given position."""
    group = self.add_option_group(*args, **kwargs)
    self.option_groups.pop()
    self.option_groups.insert(idx, group)
    return group