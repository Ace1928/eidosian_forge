import sys
import shutil
from getopt import getopt, GetoptError
import os
from os import environ, mkdir
from os.path import dirname, join, basename, exists, expanduser
import pkgutil
import re
import importlib
from kivy.logger import Logger, LOG_LEVELS
from kivy.utils import platform
from kivy._version import __version__, RELEASE as _KIVY_RELEASE, \
from kivy.logger import file_log_handler
def kivy_configure():
    """Call post-configuration of Kivy.
    This function must be called if you create the window yourself.
    """
    for callback in __kivy_post_configuration:
        callback()