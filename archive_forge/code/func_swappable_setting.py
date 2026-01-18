import functools
import inspect
import warnings
from functools import partial
from django import forms
from django.apps import apps
from django.conf import SettingsReference, settings
from django.core import checks, exceptions
from django.db import connection, router
from django.db.backends import utils
from django.db.models import Q
from django.db.models.constants import LOOKUP_SEP
from django.db.models.deletion import CASCADE, SET_DEFAULT, SET_NULL
from django.db.models.query_utils import PathInfo
from django.db.models.utils import make_model_tuple
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.functional import cached_property
from django.utils.translation import gettext_lazy as _
from . import Field
from .mixins import FieldCacheMixin
from .related_descriptors import (
from .related_lookups import (
from .reverse_related import ForeignObjectRel, ManyToManyRel, ManyToOneRel, OneToOneRel
@property
def swappable_setting(self):
    """
        Get the setting that this is powered from for swapping, or None
        if it's not swapped in / marked with swappable=False.
        """
    if self.swappable:
        if isinstance(self.remote_field.model, str):
            to_string = self.remote_field.model
        else:
            to_string = self.remote_field.model._meta.label
        return apps.get_swappable_settings_name(to_string)
    return None