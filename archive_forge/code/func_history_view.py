import copy
import enum
import json
import re
from functools import partial, update_wrapper
from urllib.parse import parse_qsl
from urllib.parse import quote as urlquote
from urllib.parse import urlparse
from django import forms
from django.conf import settings
from django.contrib import messages
from django.contrib.admin import helpers, widgets
from django.contrib.admin.checks import (
from django.contrib.admin.exceptions import DisallowedModelAdminToField, NotRegistered
from django.contrib.admin.templatetags.admin_urls import add_preserved_filters
from django.contrib.admin.utils import (
from django.contrib.admin.widgets import AutocompleteSelect, AutocompleteSelectMultiple
from django.contrib.auth import get_permission_codename
from django.core.exceptions import (
from django.core.paginator import Paginator
from django.db import models, router, transaction
from django.db.models.constants import LOOKUP_SEP
from django.forms.formsets import DELETION_FIELD_NAME, all_valid
from django.forms.models import (
from django.forms.widgets import CheckboxSelectMultiple, SelectMultiple
from django.http import HttpResponseRedirect
from django.http.response import HttpResponseBase
from django.template.response import SimpleTemplateResponse, TemplateResponse
from django.urls import reverse
from django.utils.decorators import method_decorator
from django.utils.html import format_html
from django.utils.http import urlencode
from django.utils.safestring import mark_safe
from django.utils.text import (
from django.utils.translation import gettext as _
from django.utils.translation import ngettext
from django.views.decorators.csrf import csrf_protect
from django.views.generic import RedirectView
def history_view(self, request, object_id, extra_context=None):
    """The 'history' admin view for this model."""
    from django.contrib.admin.models import LogEntry
    from django.contrib.admin.views.main import PAGE_VAR
    model = self.model
    obj = self.get_object(request, unquote(object_id))
    if obj is None:
        return self._get_obj_does_not_exist_redirect(request, model._meta, object_id)
    if not self.has_view_or_change_permission(request, obj):
        raise PermissionDenied
    app_label = self.opts.app_label
    action_list = LogEntry.objects.filter(object_id=unquote(object_id), content_type=get_content_type_for_model(model)).select_related().order_by('action_time')
    paginator = self.get_paginator(request, action_list, 100)
    page_number = request.GET.get(PAGE_VAR, 1)
    page_obj = paginator.get_page(page_number)
    page_range = paginator.get_elided_page_range(page_obj.number)
    context = {**self.admin_site.each_context(request), 'title': _('Change history: %s') % obj, 'subtitle': None, 'action_list': page_obj, 'page_range': page_range, 'page_var': PAGE_VAR, 'pagination_required': paginator.count > 100, 'module_name': str(capfirst(self.opts.verbose_name_plural)), 'object': obj, 'opts': self.opts, 'preserved_filters': self.get_preserved_filters(request), **(extra_context or {})}
    request.current_app = self.admin_site.name
    return TemplateResponse(request, self.object_history_template or ['admin/%s/%s/object_history.html' % (app_label, self.opts.model_name), 'admin/%s/object_history.html' % app_label, 'admin/object_history.html'], context)