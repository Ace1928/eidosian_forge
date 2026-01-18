import contextlib
import functools
import os
from collections import defaultdict
from functools import partial
from functools import wraps
from typing import (
from ..errors import FileError, OptionError
from ..extern.packaging.markers import default_environment as marker_env
from ..extern.packaging.requirements import InvalidRequirement, Requirement
from ..extern.packaging.specifiers import SpecifierSet
from ..extern.packaging.version import InvalidVersion, Version
from ..warnings import SetuptoolsDeprecationWarning
from . import expand
def parse_section(self, section_options):
    """Parses configuration file section.

        :param dict section_options:
        """
    for name, (_, value) in section_options.items():
        with contextlib.suppress(KeyError):
            self[name] = value