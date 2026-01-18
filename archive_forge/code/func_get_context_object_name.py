from django.core.exceptions import ImproperlyConfigured
from django.db import models
from django.http import Http404
from django.utils.translation import gettext as _
from django.views.generic.base import ContextMixin, TemplateResponseMixin, View
def get_context_object_name(self, obj):
    """Get the name to use for the object."""
    if self.context_object_name:
        return self.context_object_name
    elif isinstance(obj, models.Model):
        return obj._meta.model_name
    else:
        return None