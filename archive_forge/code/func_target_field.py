import warnings
from django.core import exceptions
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.functional import cached_property
from django.utils.hashable import make_hashable
from . import BLANK_CHOICE_DASH
from .mixins import FieldCacheMixin
@property
def target_field(self):
    """
        When filtering against this relation, return the field on the remote
        model against which the filtering should happen.
        """
    target_fields = self.path_infos[-1].target_fields
    if len(target_fields) > 1:
        raise exceptions.FieldError("Can't use target_field for multicolumn relations.")
    return target_fields[0]