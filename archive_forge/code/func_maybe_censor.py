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
def maybe_censor(key, value, mask='*' * 8):
    if isinstance(value, Mapping):
        return filter_hidden_settings(value)
    if isinstance(key, str):
        if HIDDEN_SETTINGS.search(key):
            return mask
        elif 'broker_url' in key.lower():
            from kombu import Connection
            return Connection(value).as_uri(mask=mask)
        elif 'backend' in key.lower():
            return maybe_sanitize_url(value, mask=mask)
    return value