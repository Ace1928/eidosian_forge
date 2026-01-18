import logging
import os
import shutil
import sys
import urllib.parse
from typing import (
from pip._internal.cli.spinners import SpinnerInterface
from pip._internal.exceptions import BadCommand, InstallationError
from pip._internal.utils.misc import (
from pip._internal.utils.subprocess import (
from pip._internal.utils.urls import get_url_scheme
@property
def all_schemes(self) -> List[str]:
    schemes: List[str] = []
    for backend in self.backends:
        schemes.extend(backend.schemes)
    return schemes