import functools
from django.core.exceptions import ImproperlyConfigured
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
from .base import Template
from .context import Context, _builtin_context_processors
from .exceptions import TemplateDoesNotExist
from .library import import_library
def get_template_loaders(self, template_loaders):
    loaders = []
    for template_loader in template_loaders:
        loader = self.find_template_loader(template_loader)
        if loader is not None:
            loaders.append(loader)
    return loaders