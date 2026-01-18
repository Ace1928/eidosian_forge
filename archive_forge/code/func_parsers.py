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
@property
def parsers(self):
    """Metadata item name to parser function mapping."""
    parse_list = self._parse_list
    parse_bool = self._parse_bool
    parse_dict = self._parse_dict
    parse_cmdclass = self._parse_cmdclass
    return {'zip_safe': parse_bool, 'include_package_data': parse_bool, 'package_dir': parse_dict, 'scripts': parse_list, 'eager_resources': parse_list, 'dependency_links': parse_list, 'namespace_packages': self._deprecated_config_handler(parse_list, 'The namespace_packages parameter is deprecated, consider using implicit namespaces instead (PEP 420).'), 'install_requires': partial(self._parse_requirements_list, 'install_requires'), 'setup_requires': self._parse_list_semicolon, 'tests_require': self._parse_list_semicolon, 'packages': self._parse_packages, 'entry_points': self._parse_file_in_root, 'py_modules': parse_list, 'python_requires': SpecifierSet, 'cmdclass': parse_cmdclass}