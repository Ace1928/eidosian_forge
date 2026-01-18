import logging
import os
import pathlib
import site
import sys
import textwrap
from collections import OrderedDict
from types import TracebackType
from typing import TYPE_CHECKING, Iterable, List, Optional, Set, Tuple, Type, Union
from pip._vendor.certifi import where
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.version import Version
from pip import __file__ as pip_location
from pip._internal.cli.spinners import open_spinner
from pip._internal.locations import get_platlib, get_purelib, get_scheme
from pip._internal.metadata import get_default_environment, get_environment
from pip._internal.utils.subprocess import call_subprocess
from pip._internal.utils.temp_dir import TempDirectory, tempdir_kinds
def install_requirements(self, finder: 'PackageFinder', requirements: Iterable[str], prefix_as_string: str, *, kind: str) -> None:
    raise NotImplementedError()