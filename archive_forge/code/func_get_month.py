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
def get_month(self):
    """Return the month for which this view should display data."""
    month = self.month
    if month is None:
        try:
            month = self.kwargs['month']
        except KeyError:
            try:
                month = self.request.GET['month']
            except KeyError:
                raise Http404(_('No month specified'))
    return month