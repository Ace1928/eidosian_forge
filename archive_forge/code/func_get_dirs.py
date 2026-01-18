import hashlib
from django.template import TemplateDoesNotExist
from django.template.backends.django import copy_exception
from .base import Loader as BaseLoader
def get_dirs(self):
    for loader in self.loaders:
        if hasattr(loader, 'get_dirs'):
            yield from loader.get_dirs()