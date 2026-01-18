import errno
import functools
import os
import re
import subprocess
import sys
from typing import Callable
@register_vcs_handler('git', 'get_keywords')
def git_get_keywords(versionfile_abs):
    """Extract version information from the given file."""
    keywords = {}
    try:
        with open(versionfile_abs, encoding='utf-8') as fobj:
            for line in fobj:
                if line.strip().startswith('git_refnames ='):
                    mo = re.search('=\\s*"(.*)"', line)
                    if mo:
                        keywords['refnames'] = mo.group(1)
                if line.strip().startswith('git_full ='):
                    mo = re.search('=\\s*"(.*)"', line)
                    if mo:
                        keywords['full'] = mo.group(1)
                if line.strip().startswith('git_date ='):
                    mo = re.search('=\\s*"(.*)"', line)
                    if mo:
                        keywords['date'] = mo.group(1)
    except OSError:
        pass
    return keywords