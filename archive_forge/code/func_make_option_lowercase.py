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
def make_option_lowercase(self, opt, section):
    if section != 'metadata' or opt.islower():
        return opt
    lowercase_opt = opt.lower()
    SetuptoolsDeprecationWarning.emit('Invalid uppercase configuration', f'\n            Usage of uppercase key {opt!r} in {section!r} will not be supported in\n            future versions. Please use lowercase {lowercase_opt!r} instead.\n            ', see_docs='userguide/declarative_config.html', due_date=(2024, 9, 26))
    return lowercase_opt