import functools
import sys
import threading
import warnings
from collections import Counter, defaultdict
from functools import partial
from django.core.exceptions import AppRegistryNotReady, ImproperlyConfigured
from .config import AppConfig
def get_registered_model(self, app_label, model_name):
    """
        Similar to get_model(), but doesn't require that an app exists with
        the given app_label.

        It's safe to call this method at import time, even while the registry
        is being populated.
        """
    model = self.all_models[app_label].get(model_name.lower())
    if model is None:
        raise LookupError("Model '%s.%s' not registered." % (app_label, model_name))
    return model