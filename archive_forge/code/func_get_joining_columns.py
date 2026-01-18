import warnings
from django.core import exceptions
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.functional import cached_property
from django.utils.hashable import make_hashable
from . import BLANK_CHOICE_DASH
from .mixins import FieldCacheMixin
def get_joining_columns(self):
    warnings.warn('ForeignObjectRel.get_joining_columns() is deprecated. Use get_joining_fields() instead.', RemovedInDjango60Warning)
    return self.field.get_reverse_joining_columns()