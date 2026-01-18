from pathlib import Path
import jinja2
from django.conf import settings
from django.template import TemplateDoesNotExist, TemplateSyntaxError
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
from .base import BaseEngine
from .utils import csrf_input_lazy, csrf_token_lazy
@cached_property
def template_context_processors(self):
    return [import_string(path) for path in self.context_processors]