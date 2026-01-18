import os
import time
import warnings
from asgiref.local import Local
from django.apps import apps
from django.core.exceptions import ImproperlyConfigured
from django.core.signals import setting_changed
from django.db import connections, router
from django.db.utils import ConnectionRouter
from django.dispatch import Signal, receiver
from django.utils import timezone
from django.utils.formats import FORMAT_SETTINGS, reset_format_cache
from django.utils.functional import empty
from django.utils.module_loading import import_string
@receiver(setting_changed)
def auth_password_validators_changed(*, setting, **kwargs):
    if setting == 'AUTH_PASSWORD_VALIDATORS':
        from django.contrib.auth.password_validation import get_default_password_validators
        get_default_password_validators.cache_clear()