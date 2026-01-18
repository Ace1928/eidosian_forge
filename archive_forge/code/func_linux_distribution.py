import argparse
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import warnings
from typing import (
def linux_distribution(self, full_distribution_name: bool=True) -> Tuple[str, str, str]:
    """
        Return information about the OS distribution that is compatible
        with Python's :func:`platform.linux_distribution`, supporting a subset
        of its parameters.

        For details, see :func:`distro.linux_distribution`.
        """
    return (self.name() if full_distribution_name else self.id(), self.version(), self._os_release_info.get('release_codename') or self.codename())