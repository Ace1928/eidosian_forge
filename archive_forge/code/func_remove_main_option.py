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
def remove_main_option(self, name: str) -> None:
    self.file_config.remove_option(self.config_ini_section, name)