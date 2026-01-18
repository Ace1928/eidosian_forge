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
def get_day(self):
    """Return the day for which this view should display data."""
    day = self.day
    if day is None:
        try:
            day = self.kwargs['day']
        except KeyError:
            try:
                day = self.request.GET['day']
            except KeyError:
                raise Http404(_('No day specified'))
    return day