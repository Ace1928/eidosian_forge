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
def response_action(self, request, queryset):
    """
        Handle an admin action. This is called if a request is POSTed to the
        changelist; it returns an HttpResponse if the action was handled, and
        None otherwise.
        """
    try:
        action_index = int(request.POST.get('index', 0))
    except ValueError:
        action_index = 0
    data = request.POST.copy()
    data.pop(helpers.ACTION_CHECKBOX_NAME, None)
    data.pop('index', None)
    try:
        data.update({'action': data.getlist('action')[action_index]})
    except IndexError:
        pass
    action_form = self.action_form(data, auto_id=None)
    action_form.fields['action'].choices = self.get_action_choices(request)
    if action_form.is_valid():
        action = action_form.cleaned_data['action']
        select_across = action_form.cleaned_data['select_across']
        func = self.get_actions(request)[action][0]
        selected = request.POST.getlist(helpers.ACTION_CHECKBOX_NAME)
        if not selected and (not select_across):
            msg = _('Items must be selected in order to perform actions on them. No items have been changed.')
            self.message_user(request, msg, messages.WARNING)
            return None
        if not select_across:
            queryset = queryset.filter(pk__in=selected)
        response = func(self, request, queryset)
        if isinstance(response, HttpResponseBase):
            return response
        else:
            return HttpResponseRedirect(request.get_full_path())
    else:
        msg = _('No action selected.')
        self.message_user(request, msg, messages.WARNING)
        return None