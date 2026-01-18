from importlib import import_module
from pkgutil import walk_packages
from django.apps import apps
from django.conf import settings
from django.template import TemplateDoesNotExist
from django.template.context import make_context
from django.template.engine import Engine
from django.template.library import InvalidTemplateLibrary
from .base import BaseEngine
def get_package_libraries(pkg):
    """
    Recursively yield template tag libraries defined in submodules of a
    package.
    """
    for entry in walk_packages(pkg.__path__, pkg.__name__ + '.'):
        try:
            module = import_module(entry[1])
        except ImportError as e:
            raise InvalidTemplateLibrary("Invalid template library specified. ImportError raised when trying to load '%s': %s" % (entry[1], e)) from e
        if hasattr(module, 'register'):
            yield entry[1]