import functools
import sys
import threading
import warnings
from collections import Counter, defaultdict
from functools import partial
from django.core.exceptions import AppRegistryNotReady, ImproperlyConfigured
from .config import AppConfig
def get_containing_app_config(self, object_name):
    """
        Look for an app config containing a given object.

        object_name is the dotted Python path to the object.

        Return the app config for the inner application in case of nesting.
        Return None if the object isn't in any registered app config.
        """
    self.check_apps_ready()
    candidates = []
    for app_config in self.app_configs.values():
        if object_name.startswith(app_config.name):
            subpath = object_name.removeprefix(app_config.name)
            if subpath == '' or subpath[0] == '.':
                candidates.append(app_config)
    if candidates:
        return sorted(candidates, key=lambda ac: -len(ac.name))[0]