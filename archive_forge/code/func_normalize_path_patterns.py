import fnmatch
import os
import shutil
import subprocess
from pathlib import Path
from subprocess import run
from django.apps import apps as installed_apps
from django.utils.crypto import get_random_string
from django.utils.encoding import DEFAULT_LOCALE_ENCODING
from .base import CommandError, CommandParser
def normalize_path_patterns(patterns):
    """Normalize an iterable of glob style patterns based on OS."""
    patterns = [os.path.normcase(p) for p in patterns]
    dir_suffixes = {'%s*' % path_sep for path_sep in {'/', os.sep}}
    norm_patterns = []
    for pattern in patterns:
        for dir_suffix in dir_suffixes:
            if pattern.endswith(dir_suffix):
                norm_patterns.append(pattern.removesuffix(dir_suffix))
                break
        else:
            norm_patterns.append(pattern)
    return norm_patterns