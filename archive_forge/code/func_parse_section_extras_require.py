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
def parse_section_extras_require(self, section_options):
    """Parses `extras_require` configuration file section.

        :param dict section_options:
        """
    parsed = self._parse_section_to_dict_with_key(section_options, lambda k, v: self._parse_requirements_list(f'extras_require[{k}]', v))
    self['extras_require'] = parsed