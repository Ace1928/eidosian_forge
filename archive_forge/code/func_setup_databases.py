import collections
import logging
import os
import re
import sys
import time
import warnings
from contextlib import contextmanager
from functools import wraps
from io import StringIO
from itertools import chain
from types import SimpleNamespace
from unittest import TestCase, skipIf, skipUnless
from xml.dom.minidom import Node, parseString
from asgiref.sync import iscoroutinefunction
from django.apps import apps
from django.apps.registry import Apps
from django.conf import UserSettingsHolder, settings
from django.core import mail
from django.core.exceptions import ImproperlyConfigured
from django.core.signals import request_started, setting_changed
from django.db import DEFAULT_DB_ALIAS, connections, reset_queries
from django.db.models.options import Options
from django.template import Template
from django.test.signals import template_rendered
from django.urls import get_script_prefix, set_script_prefix
from django.utils.translation import deactivate
def setup_databases(verbosity, interactive, *, time_keeper=None, keepdb=False, debug_sql=False, parallel=0, aliases=None, serialized_aliases=None, **kwargs):
    """Create the test databases."""
    if time_keeper is None:
        time_keeper = NullTimeKeeper()
    test_databases, mirrored_aliases = get_unique_databases_and_mirrors(aliases)
    old_names = []
    for db_name, aliases in test_databases.values():
        first_alias = None
        for alias in aliases:
            connection = connections[alias]
            old_names.append((connection, db_name, first_alias is None))
            if first_alias is None:
                first_alias = alias
                with time_keeper.timed("  Creating '%s'" % alias):
                    serialize_alias = serialized_aliases is None or alias in serialized_aliases
                    connection.creation.create_test_db(verbosity=verbosity, autoclobber=not interactive, keepdb=keepdb, serialize=serialize_alias)
                if parallel > 1:
                    for index in range(parallel):
                        with time_keeper.timed("  Cloning '%s'" % alias):
                            connection.creation.clone_test_db(suffix=str(index + 1), verbosity=verbosity, keepdb=keepdb)
            else:
                connections[alias].creation.set_as_test_mirror(connections[first_alias].settings_dict)
    for alias, mirror_alias in mirrored_aliases.items():
        connections[alias].creation.set_as_test_mirror(connections[mirror_alias].settings_dict)
    if debug_sql:
        for alias in connections:
            connections[alias].force_debug_cursor = True
    return old_names