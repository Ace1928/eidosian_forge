import datetime
from django.contrib.admin.exceptions import NotRegistered
from django.contrib.admin.options import IncorrectLookupParameters
from django.contrib.admin.utils import (
from django.core.exceptions import ImproperlyConfigured, ValidationError
from django.db import models
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
def field_admin_ordering(self, field, request, model_admin):
    """
        Return the model admin's ordering for related field, if provided.
        """
    try:
        related_admin = model_admin.admin_site.get_model_admin(field.remote_field.model)
    except NotRegistered:
        return ()
    else:
        return related_admin.get_ordering(request)