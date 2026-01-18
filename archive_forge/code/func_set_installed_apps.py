import functools
import sys
import threading
import warnings
from collections import Counter, defaultdict
from functools import partial
from django.core.exceptions import AppRegistryNotReady, ImproperlyConfigured
from .config import AppConfig
def set_installed_apps(self, installed):
    """
        Enable a different set of installed apps for get_app_config[s].

        installed must be an iterable in the same format as INSTALLED_APPS.

        set_installed_apps() must be balanced with unset_installed_apps(),
        even if it exits with an exception.

        Primarily used as a receiver of the setting_changed signal in tests.

        This method may trigger new imports, which may add new models to the
        registry of all imported models. They will stay in the registry even
        after unset_installed_apps(). Since it isn't possible to replay
        imports safely (e.g. that could lead to registering listeners twice),
        models are registered when they're imported and never removed.
        """
    if not self.ready:
        raise AppRegistryNotReady("App registry isn't ready yet.")
    self.stored_app_configs.append(self.app_configs)
    self.app_configs = {}
    self.apps_ready = self.models_ready = self.loading = self.ready = False
    self.clear_cache()
    self.populate(installed)