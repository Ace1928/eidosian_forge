import functools
import logging
import re
from typing import NewType, Optional, Tuple, cast
from pip._vendor.packaging import specifiers, version
from pip._vendor.packaging.requirements import Requirement
def check_requires_python(requires_python: Optional[str], version_info: Tuple[int, ...]) -> bool:
    """
    Check if the given Python version matches a "Requires-Python" specifier.

    :param version_info: A 3-tuple of ints representing a Python
        major-minor-micro version to check (e.g. `sys.version_info[:3]`).

    :return: `True` if the given Python version satisfies the requirement.
        Otherwise, return `False`.

    :raises InvalidSpecifier: If `requires_python` has an invalid format.
    """
    if requires_python is None:
        return True
    requires_python_specifier = specifiers.SpecifierSet(requires_python)
    python_version = version.parse('.'.join(map(str, version_info)))
    return python_version in requires_python_specifier