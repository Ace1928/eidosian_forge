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
def fetch_command(self, subcommand):
    """
        Try to fetch the given subcommand, printing a message with the
        appropriate command called from the command line (usually
        "django-admin" or "manage.py") if it can't be found.
        """
    commands = get_commands()
    try:
        app_name = commands[subcommand]
    except KeyError:
        if os.environ.get('DJANGO_SETTINGS_MODULE'):
            settings.INSTALLED_APPS
        elif not settings.configured:
            sys.stderr.write('No Django settings specified.\n')
        possible_matches = get_close_matches(subcommand, commands)
        sys.stderr.write('Unknown command: %r' % subcommand)
        if possible_matches:
            sys.stderr.write('. Did you mean %s?' % possible_matches[0])
        sys.stderr.write("\nType '%s help' for usage.\n" % self.prog_name)
        sys.exit(1)
    if isinstance(app_name, BaseCommand):
        klass = app_name
    else:
        klass = load_command_class(app_name, subcommand)
    return klass