from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import platform
import re
import subprocess
import sys
from googlecloudsdk.core.util import encoding
@property
def clean_version(self):
    """Returns a cleaned version of the operating system version."""
    version = self.version
    if self == OperatingSystem.WINDOWS:
        capitalized = version.upper()
        if capitalized in ('XP', 'VISTA'):
            return version
        if capitalized.startswith('SERVER'):
            return version[:11].replace(' ', '_')
    matches = re.match('(\\d+)(\\.\\d+)?(\\.\\d+)?.*', version)
    if not matches:
        return None
    return ''.join((group for group in matches.groups() if group))