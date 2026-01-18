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
@util.memoized_property
def file_config(self) -> ConfigParser:
    """Return the underlying ``ConfigParser`` object.

        Direct access to the .ini file is available here,
        though the :meth:`.Config.get_section` and
        :meth:`.Config.get_main_option`
        methods provide a possibly simpler interface.

        """
    if self.config_file_name:
        here = os.path.abspath(os.path.dirname(self.config_file_name))
    else:
        here = ''
    self.config_args['here'] = here
    file_config = ConfigParser(self.config_args)
    if self.config_file_name:
        compat.read_config_parser(file_config, [self.config_file_name])
    else:
        file_config.add_section(self.config_ini_section)
    return file_config