from __future__ import annotations
from argparse import ArgumentParser
from argparse import Namespace
from configparser import ConfigParser
import inspect
import os
import sys
from typing import Any
from typing import cast
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import TextIO
from typing import Union
from typing_extensions import TypedDict
from . import __version__
from . import command
from . import util
from .util import compat
def get_main_option(self, name: str, default: Optional[str]=None) -> Optional[str]:
    """Return an option from the 'main' section of the .ini file.

        This defaults to being a key from the ``[alembic]``
        section, unless the ``-n/--name`` flag were used to
        indicate a different section.

        """
    return self.get_section_option(self.config_ini_section, name, default)