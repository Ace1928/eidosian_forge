import functools
import os
import pkgutil
import sys
from argparse import (
from collections import defaultdict
from difflib import get_close_matches
from importlib import import_module
import django
from django.apps import apps
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.core.management.base import (
from django.core.management.color import color_style
from django.utils import autoreload
@functools.cache
def get_commands():
    """
    Return a dictionary mapping command names to their callback applications.

    Look for a management.commands package in django.core, and in each
    installed application -- if a commands package exists, register all
    commands in that package.

    Core commands are always included. If a settings module has been
    specified, also include user-defined commands.

    The dictionary is in the format {command_name: app_name}. Key-value
    pairs from this dictionary can then be used in calls to
    load_command_class(app_name, command_name)

    The dictionary is cached on the first call and reused on subsequent
    calls.
    """
    commands = {name: 'django.core' for name in find_commands(__path__[0])}
    if not settings.configured:
        return commands
    for app_config in reversed(apps.get_app_configs()):
        path = os.path.join(app_config.path, 'management')
        commands.update({name: app_config.name for name in find_commands(path)})
    return commands