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
def parse_section_packages__find(self, section_options):
    """Parses `packages.find` configuration file section.

        To be used in conjunction with _parse_packages().

        :param dict section_options:
        """
    section_data = self._parse_section_to_dict(section_options, self._parse_list)
    valid_keys = ['where', 'include', 'exclude']
    find_kwargs = dict([(k, v) for k, v in section_data.items() if k in valid_keys and v])
    where = find_kwargs.get('where')
    if where is not None:
        find_kwargs['where'] = where[0]
    return find_kwargs