import pandas.util._test_decorators as td
from __future__ import annotations
import locale
from typing import (
import pytest
from pandas._config import get_option
from pandas._config.config import _get_option
from pandas.compat import (
from pandas.compat._optional import import_optional_dependency
def skip_if_installed(package: str) -> pytest.MarkDecorator:
    """
    Skip a test if a package is installed.

    Parameters
    ----------
    package : str
        The name of the package.

    Returns
    -------
    pytest.MarkDecorator
        a pytest.mark.skipif to use as either a test decorator or a
        parametrization mark.
    """
    return pytest.mark.skipif(bool(import_optional_dependency(package, errors='ignore')), reason=f'Skipping because {package} is installed.')