import argparse
import io
import logging
import os
import platform
import re
import shutil
import sys
import subprocess
from . import envvar
from .deprecation import deprecated
from .errors import DeveloperError
import pyomo.common
from pyomo.common.dependencies import attempt_import
def get_platform_url(self, urlmap):
    """Select the url for this platform

        Given a `urlmap` dict that maps the platform name (from
        `FileDownloader.get_sysinfo()`) to a platform-specific URL,
        return the URL that matches the current platform.

        Parameters
        ----------
        urlmap: dict
            Map of platform name (e.g., `linux`, `windows`, `cygwin`,
            `darwin`) to URL

        """
    system, bits = self.get_sysinfo()
    url = urlmap.get(system, None)
    if url is None:
        raise RuntimeError("cannot infer the correct url for platform '%s'" % (platform,))
    return url