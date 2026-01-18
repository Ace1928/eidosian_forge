from django.core.exceptions import ImproperlyConfigured
from django.db import models
from django.http import Http404
from django.utils.translation import gettext as _
from django.views.generic.base import ContextMixin, TemplateResponseMixin, View
def get_slug_field(self):
    """Get the name of a slug field to be used to look up by slug."""
    return self.slug_field