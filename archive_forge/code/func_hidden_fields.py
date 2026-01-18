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
def hidden_fields(self):
    """
        Return a list of all the BoundField objects that are hidden fields.
        Useful for manual form layout in templates.
        """
    return [field for field in self if field.is_hidden]