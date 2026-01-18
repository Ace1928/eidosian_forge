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
def teardown_databases(old_config, verbosity, parallel=0, keepdb=False):
    """Destroy all the non-mirror databases."""
    for connection, old_name, destroy in old_config:
        if destroy:
            if parallel > 1:
                for index in range(parallel):
                    connection.creation.destroy_test_db(suffix=str(index + 1), verbosity=verbosity, keepdb=keepdb)
            connection.creation.destroy_test_db(old_name, verbosity, keepdb)