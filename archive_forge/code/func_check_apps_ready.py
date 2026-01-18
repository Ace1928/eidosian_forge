import functools
import sys
import threading
import warnings
from collections import Counter, defaultdict
from functools import partial
from django.core.exceptions import AppRegistryNotReady, ImproperlyConfigured
from .config import AppConfig
def check_apps_ready(self):
    """Raise an exception if all apps haven't been imported yet."""
    if not self.apps_ready:
        from django.conf import settings
        settings.INSTALLED_APPS
        raise AppRegistryNotReady("Apps aren't loaded yet.")