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
def get_unique_databases_and_mirrors(aliases=None):
    """
    Figure out which databases actually need to be created.

    Deduplicate entries in DATABASES that correspond the same database or are
    configured as test mirrors.

    Return two values:
    - test_databases: ordered mapping of signatures to (name, list of aliases)
                      where all aliases share the same underlying database.
    - mirrored_aliases: mapping of mirror aliases to original aliases.
    """
    if aliases is None:
        aliases = connections
    mirrored_aliases = {}
    test_databases = {}
    dependencies = {}
    default_sig = connections[DEFAULT_DB_ALIAS].creation.test_db_signature()
    for alias in connections:
        connection = connections[alias]
        test_settings = connection.settings_dict['TEST']
        if test_settings['MIRROR']:
            mirrored_aliases[alias] = test_settings['MIRROR']
        elif alias in aliases:
            item = test_databases.setdefault(connection.creation.test_db_signature(), (connection.settings_dict['NAME'], []))
            if alias == DEFAULT_DB_ALIAS:
                item[1].insert(0, alias)
            else:
                item[1].append(alias)
            if 'DEPENDENCIES' in test_settings:
                dependencies[alias] = test_settings['DEPENDENCIES']
            elif alias != DEFAULT_DB_ALIAS and connection.creation.test_db_signature() != default_sig:
                dependencies[alias] = test_settings.get('DEPENDENCIES', [DEFAULT_DB_ALIAS])
    test_databases = dict(dependency_ordered(test_databases.items(), dependencies))
    return (test_databases, mirrored_aliases)