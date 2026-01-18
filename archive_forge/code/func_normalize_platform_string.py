from __future__ import (absolute_import, division, print_function)
import re
def normalize_platform_string(string, daemon_os=None, daemon_arch=None):
    return str(_Platform.parse_platform_string(string, daemon_os=daemon_os, daemon_arch=daemon_arch))