import functools
import os
from django.apps import apps
from django.conf import settings
from django.contrib.staticfiles import utils
from django.core.checks import Error, Warning
from django.core.exceptions import ImproperlyConfigured
from django.core.files.storage import FileSystemStorage, Storage, default_storage
from django.utils._os import safe_join
from django.utils.functional import LazyObject, empty
from django.utils.module_loading import import_string
@functools.cache
def get_finder(import_path):
    """
    Import the staticfiles finder class described by import_path, where
    import_path is the full Python path to the class.
    """
    Finder = import_string(import_path)
    if not issubclass(Finder, BaseFinder):
        raise ImproperlyConfigured('Finder "%s" is not a subclass of "%s"' % (Finder, BaseFinder))
    return Finder()