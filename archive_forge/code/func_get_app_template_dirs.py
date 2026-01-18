import functools
from collections import Counter
from pathlib import Path
from django.apps import apps
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
@functools.lru_cache
def get_app_template_dirs(dirname):
    """
    Return an iterable of paths of directories to load app templates from.

    dirname is the name of the subdirectory containing templates inside
    installed applications.
    """
    template_dirs = [Path(app_config.path) / dirname for app_config in apps.get_app_configs() if app_config.path and (Path(app_config.path) / dirname).is_dir()]
    return tuple(template_dirs)