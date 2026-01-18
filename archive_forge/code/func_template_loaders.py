import functools
from django.core.exceptions import ImproperlyConfigured
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
from .base import Template
from .context import Context, _builtin_context_processors
from .exceptions import TemplateDoesNotExist
from .library import import_library
@cached_property
def template_loaders(self):
    return self.get_template_loaders(self.loaders)