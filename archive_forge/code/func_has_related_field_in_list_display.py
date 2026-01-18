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
def has_related_field_in_list_display(self):
    for field_name in self.list_display:
        try:
            field = self.lookup_opts.get_field(field_name)
        except FieldDoesNotExist:
            pass
        else:
            if isinstance(field.remote_field, ManyToOneRel):
                if field_name != field.get_attname():
                    return True
    return False