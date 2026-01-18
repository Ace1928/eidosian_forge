import hashlib
from django.template import TemplateDoesNotExist
from django.template.backends.django import copy_exception
from .base import Loader as BaseLoader
def generate_hash(self, values):
    return hashlib.sha1('|'.join(values).encode()).hexdigest()