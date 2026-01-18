from django.core.exceptions import ImproperlyConfigured
from django.core.paginator import InvalidPage, Paginator
from django.db.models import QuerySet
from django.http import Http404
from django.utils.translation import gettext as _
from django.views.generic.base import ContextMixin, TemplateResponseMixin, View
def get_paginate_by(self, queryset):
    """
        Get the number of items to paginate by, or ``None`` for no pagination.
        """
    return self.paginate_by