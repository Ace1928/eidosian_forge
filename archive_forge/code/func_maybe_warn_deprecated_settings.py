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
def maybe_warn_deprecated_settings(self):
    if self.deprecated_settings:
        from celery.app.defaults import _TO_NEW_KEY
        from celery.utils import deprecated
        for setting in self.deprecated_settings:
            deprecated.warn(description=f'The {setting!r} setting', removal='6.0.0', alternative=f'Use the {_TO_NEW_KEY[setting]} instead')
        return True
    return False