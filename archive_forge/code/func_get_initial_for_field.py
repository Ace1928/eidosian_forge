import copy
import datetime
from django.core.exceptions import NON_FIELD_ERRORS, ValidationError
from django.forms.fields import Field, FileField
from django.forms.utils import ErrorDict, ErrorList, RenderableFormMixin
from django.forms.widgets import Media, MediaDefiningClass
from django.utils.datastructures import MultiValueDict
from django.utils.functional import cached_property
from django.utils.translation import gettext as _
from .renderers import get_default_renderer
def get_initial_for_field(self, field, field_name):
    """
        Return initial data for field on form. Use initial data from the form
        or the field, in that order. Evaluate callable values.
        """
    value = self.initial.get(field_name, field.initial)
    if callable(value):
        value = value()
    if isinstance(value, (datetime.datetime, datetime.time)) and (not field.widget.supports_microseconds):
        value = value.replace(microsecond=0)
    return value