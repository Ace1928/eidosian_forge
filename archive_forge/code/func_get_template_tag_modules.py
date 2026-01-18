from importlib import import_module
from pkgutil import walk_packages
from django.apps import apps
from django.conf import settings
from django.template import TemplateDoesNotExist
from django.template.context import make_context
from django.template.engine import Engine
from django.template.library import InvalidTemplateLibrary
from .base import BaseEngine
def get_template_tag_modules():
    """
    Yield (module_name, module_path) pairs for all installed template tag
    libraries.
    """
    candidates = ['django.templatetags']
    candidates.extend((f'{app_config.name}.templatetags' for app_config in apps.get_app_configs()))
    for candidate in candidates:
        try:
            pkg = import_module(candidate)
        except ImportError:
            continue
        if hasattr(pkg, '__path__'):
            for name in get_package_libraries(pkg):
                yield (name.removeprefix(candidate).lstrip('.'), name)