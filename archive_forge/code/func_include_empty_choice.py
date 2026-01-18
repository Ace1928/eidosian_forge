import datetime
from django.contrib.admin.exceptions import NotRegistered
from django.contrib.admin.options import IncorrectLookupParameters
from django.contrib.admin.utils import (
from django.core.exceptions import ImproperlyConfigured, ValidationError
from django.db import models
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
@property
def include_empty_choice(self):
    """
        Return True if a "(None)" choice should be included, which filters
        out everything except empty relationships.
        """
    return self.field.null or (self.field.is_relation and self.field.many_to_many)