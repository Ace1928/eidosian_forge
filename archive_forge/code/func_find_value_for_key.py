import os
import platform as _platform
import re
from collections import namedtuple
from collections.abc import Mapping
from copy import deepcopy
from types import ModuleType
from kombu.utils.url import maybe_sanitize_url
from celery.exceptions import ImproperlyConfigured
from celery.platforms import pyimplementation
from celery.utils.collections import ConfigurationView
from celery.utils.imports import import_from_cwd, qualname, symbol_by_name
from celery.utils.text import pretty
from .defaults import _OLD_DEFAULTS, _OLD_SETTING_KEYS, _TO_NEW_KEY, _TO_OLD_KEY, DEFAULTS, SETTING_KEYS, find
def find_value_for_key(self, name, namespace='celery'):
    """Shortcut to ``get_by_parts(*find_option(name)[:-1])``."""
    return self.get_by_parts(*self.find_option(name, namespace)[:-1])