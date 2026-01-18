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
def get_random_secret_key():
    """
    Return a 50 character random string usable as a SECRET_KEY setting value.
    """
    chars = 'abcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*(-_=+)'
    return get_random_string(50, chars)