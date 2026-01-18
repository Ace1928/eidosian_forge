import json
import warnings
from django import forms
from django.core import checks, exceptions
from django.db import NotSupportedError, connections, router
from django.db.models import expressions, lookups
from django.db.models.constants import LOOKUP_SEP
from django.db.models.fields import TextField
from django.db.models.lookups import (
from django.utils.deprecation import RemovedInDjango51Warning
from django.utils.translation import gettext_lazy as _
from . import Field
from .mixins import CheckFieldDefaultMixin
def preprocess_lhs(self, compiler, connection):
    key_transforms = [self.key_name]
    previous = self.lhs
    while isinstance(previous, KeyTransform):
        key_transforms.insert(0, previous.key_name)
        previous = previous.lhs
    lhs, params = compiler.compile(previous)
    if connection.vendor == 'oracle':
        key_transforms = [key.replace('%', '%%') for key in key_transforms]
    return (lhs, params, key_transforms)