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
@classmethod
def from_lookup(cls, lookup):
    transform, *keys = lookup.split(LOOKUP_SEP)
    if not keys:
        raise ValueError('Lookup must contain key or index transforms.')
    for key in keys:
        transform = cls(key, transform)
    return transform