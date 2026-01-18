import datetime
from django.contrib.admin.exceptions import NotRegistered
from django.contrib.admin.options import IncorrectLookupParameters
from django.contrib.admin.utils import (
from django.core.exceptions import ImproperlyConfigured, ValidationError
from django.db import models
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
def field_choices(self, field, request, model_admin):
    pk_qs = model_admin.get_queryset(request).distinct().values_list('%s__pk' % self.field_path, flat=True)
    ordering = self.field_admin_ordering(field, request, model_admin)
    return field.get_choices(include_blank=False, limit_choices_to={'pk__in': pk_qs}, ordering=ordering)