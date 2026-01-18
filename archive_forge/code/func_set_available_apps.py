import functools
import sys
import threading
import warnings
from collections import Counter, defaultdict
from functools import partial
from django.core.exceptions import AppRegistryNotReady, ImproperlyConfigured
from .config import AppConfig
def set_available_apps(self, available):
    """
        Restrict the set of installed apps used by get_app_config[s].

        available must be an iterable of application names.

        set_available_apps() must be balanced with unset_available_apps().

        Primarily used for performance optimization in TransactionTestCase.

        This method is safe in the sense that it doesn't trigger any imports.
        """
    available = set(available)
    installed = {app_config.name for app_config in self.get_app_configs()}
    if not available.issubset(installed):
        raise ValueError("Available apps isn't a subset of installed apps, extra apps: %s" % ', '.join(available - installed))
    self.stored_app_configs.append(self.app_configs)
    self.app_configs = {label: app_config for label, app_config in self.app_configs.items() if app_config.name in available}
    self.clear_cache()