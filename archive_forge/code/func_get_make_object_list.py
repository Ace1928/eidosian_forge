import datetime
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.db import models
from django.http import Http404
from django.utils import timezone
from django.utils.functional import cached_property
from django.utils.translation import gettext as _
from django.views.generic.base import View
from django.views.generic.detail import (
from django.views.generic.list import (
def get_make_object_list(self):
    """
        Return `True` if this view should contain the full list of objects in
        the given year.
        """
    return self.make_object_list