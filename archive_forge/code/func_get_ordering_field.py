import warnings
from datetime import datetime, timedelta
from django import forms
from django.conf import settings
from django.contrib import messages
from django.contrib.admin import FieldListFilter
from django.contrib.admin.exceptions import (
from django.contrib.admin.options import (
from django.contrib.admin.utils import (
from django.core.exceptions import (
from django.core.paginator import InvalidPage
from django.db.models import F, Field, ManyToOneRel, OrderBy
from django.db.models.expressions import Combinable
from django.urls import reverse
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.http import urlencode
from django.utils.inspect import func_supports_parameter
from django.utils.timezone import make_aware
from django.utils.translation import gettext
def get_ordering_field(self, field_name):
    """
        Return the proper model field name corresponding to the given
        field_name to use for ordering. field_name may either be the name of a
        proper model field or the name of a method (on the admin or model) or a
        callable with the 'admin_order_field' attribute. Return None if no
        proper model field name can be matched.
        """
    try:
        field = self.lookup_opts.get_field(field_name)
        return field.name
    except FieldDoesNotExist:
        if callable(field_name):
            attr = field_name
        elif hasattr(self.model_admin, field_name):
            attr = getattr(self.model_admin, field_name)
        else:
            attr = getattr(self.model, field_name)
        if isinstance(attr, property) and hasattr(attr, 'fget'):
            attr = attr.fget
        return getattr(attr, 'admin_order_field', None)