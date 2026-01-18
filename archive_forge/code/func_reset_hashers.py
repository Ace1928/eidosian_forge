import base64
import binascii
import functools
import hashlib
import importlib
import math
import warnings
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.core.signals import setting_changed
from django.dispatch import receiver
from django.utils.crypto import (
from django.utils.deprecation import RemovedInDjango51Warning
from django.utils.module_loading import import_string
from django.utils.translation import gettext_noop as _
@receiver(setting_changed)
def reset_hashers(*, setting, **kwargs):
    if setting == 'PASSWORD_HASHERS':
        get_hashers.cache_clear()
        get_hashers_by_algorithm.cache_clear()