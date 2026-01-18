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
def main_help_text(self, commands_only=False):
    """Return the script's main help text, as a string."""
    if commands_only:
        usage = sorted(get_commands())
    else:
        usage = ['', "Type '%s help <subcommand>' for help on a specific subcommand." % self.prog_name, '', 'Available subcommands:']
        commands_dict = defaultdict(lambda: [])
        for name, app in get_commands().items():
            if app == 'django.core':
                app = 'django'
            else:
                app = app.rpartition('.')[-1]
            commands_dict[app].append(name)
        style = color_style()
        for app in sorted(commands_dict):
            usage.append('')
            usage.append(style.NOTICE('[%s]' % app))
            for name in sorted(commands_dict[app]):
                usage.append('    %s' % name)
        if self.settings_exception is not None:
            usage.append(style.NOTICE('Note that only Django core commands are listed as settings are not properly configured (error: %s).' % self.settings_exception))
    return '\n'.join(usage)