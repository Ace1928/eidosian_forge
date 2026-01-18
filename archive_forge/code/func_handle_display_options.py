import io
import itertools
import numbers
import os
import re
import sys
from contextlib import suppress
from glob import iglob
from pathlib import Path
from typing import List, Optional, Set
import distutils.cmd
import distutils.command
import distutils.core
import distutils.dist
import distutils.log
from distutils.debug import DEBUG
from distutils.errors import DistutilsOptionError, DistutilsSetupError
from distutils.fancy_getopt import translate_longopt
from distutils.util import strtobool
from .extern.more_itertools import partition, unique_everseen
from .extern.ordered_set import OrderedSet
from .extern.packaging.markers import InvalidMarker, Marker
from .extern.packaging.specifiers import InvalidSpecifier, SpecifierSet
from .extern.packaging.version import Version
from . import _entry_points
from . import _normalization
from . import _reqs
from . import command as _  # noqa  -- imported for side-effects
from ._importlib import metadata
from .config import setupcfg, pyprojecttoml
from .discovery import ConfigDiscovery
from .monkey import get_unpatched
from .warnings import InformationOnly, SetuptoolsDeprecationWarning
def handle_display_options(self, option_order):
    """If there were any non-global "display-only" options
        (--help-commands or the metadata display options) on the command
        line, display the requested info and return true; else return
        false.
        """
    import sys
    if self.help_commands:
        return _Distribution.handle_display_options(self, option_order)
    if not isinstance(sys.stdout, io.TextIOWrapper):
        return _Distribution.handle_display_options(self, option_order)
    if sys.stdout.encoding.lower() in ('utf-8', 'utf8'):
        return _Distribution.handle_display_options(self, option_order)
    encoding = sys.stdout.encoding
    sys.stdout.reconfigure(encoding='utf-8')
    try:
        return _Distribution.handle_display_options(self, option_order)
    finally:
        sys.stdout.reconfigure(encoding=encoding)