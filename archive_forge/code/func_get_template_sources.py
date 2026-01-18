import hashlib
from django.template import TemplateDoesNotExist
from django.template.backends.django import copy_exception
from .base import Loader as BaseLoader
def get_template_sources(self, template_name):
    for loader in self.loaders:
        yield from loader.get_template_sources(template_name)