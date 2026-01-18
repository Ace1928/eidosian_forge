import datetime
from django.contrib.admin.exceptions import NotRegistered
from django.contrib.admin.options import IncorrectLookupParameters
from django.contrib.admin.utils import (
from django.core.exceptions import ImproperlyConfigured, ValidationError
from django.db import models
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
def has_output(self):
    if self.include_empty_choice:
        extra = 1
    else:
        extra = 0
    return len(self.lookup_choices) + extra > 1