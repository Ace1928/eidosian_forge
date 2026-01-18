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
def get_egg_cache_dir(self):
    from . import windows_support
    egg_cache_dir = os.path.join(os.curdir, '.eggs')
    if not os.path.exists(egg_cache_dir):
        os.mkdir(egg_cache_dir)
        windows_support.hide_file(egg_cache_dir)
        readme_txt_filename = os.path.join(egg_cache_dir, 'README.txt')
        with open(readme_txt_filename, 'w') as f:
            f.write('This directory contains eggs that were downloaded by setuptools to build, test, and run plug-ins.\n\n')
            f.write('This directory caches those eggs to prevent repeated downloads.\n\n')
            f.write('However, it is safe to delete this directory.\n\n')
    return egg_cache_dir