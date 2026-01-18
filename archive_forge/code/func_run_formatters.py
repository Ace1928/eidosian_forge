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
def run_formatters(written_files, black_path=(sentinel := object())):
    """
    Run the black formatter on the specified files.
    """
    if black_path is sentinel:
        black_path = shutil.which('black')
    if black_path:
        subprocess.run([black_path, '--fast', '--', *written_files], capture_output=True)