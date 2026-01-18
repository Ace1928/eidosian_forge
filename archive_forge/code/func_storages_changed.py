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
def storages_changed(*, setting, **kwargs):
    from django.contrib.staticfiles.storage import staticfiles_storage
    from django.core.files.storage import default_storage, storages
    if setting in ('STORAGES', 'STATIC_ROOT', 'STATIC_URL'):
        try:
            del storages.backends
        except AttributeError:
            pass
        storages._backends = None
        storages._storages = {}
        default_storage._wrapped = empty
        staticfiles_storage._wrapped = empty