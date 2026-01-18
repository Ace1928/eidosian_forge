import functools
from django.core.exceptions import ImproperlyConfigured
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
from .base import Template
from .context import Context, _builtin_context_processors
from .exceptions import TemplateDoesNotExist
from .library import import_library
def find_template_loader(self, loader):
    if isinstance(loader, (tuple, list)):
        loader, *args = loader
    else:
        args = []
    if isinstance(loader, str):
        loader_class = import_string(loader)
        return loader_class(self, *args)
    else:
        raise ImproperlyConfigured('Invalid value in template loaders configuration: %r' % loader)