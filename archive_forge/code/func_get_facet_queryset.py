import datetime
from django.contrib.admin.exceptions import NotRegistered
from django.contrib.admin.options import IncorrectLookupParameters
from django.contrib.admin.utils import (
from django.core.exceptions import ImproperlyConfigured, ValidationError
from django.db import models
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
def get_facet_queryset(self, changelist):
    filtered_qs = changelist.get_queryset(self.request, exclude_parameters=self.expected_parameters())
    return filtered_qs.aggregate(**self.get_facet_counts(changelist.pk_attname, filtered_qs))