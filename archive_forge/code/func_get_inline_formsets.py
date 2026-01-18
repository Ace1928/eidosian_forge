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
def get_inline_formsets(self, request, formsets, inline_instances, obj=None):
    can_edit_parent = self.has_change_permission(request, obj) if obj else self.has_add_permission(request)
    inline_admin_formsets = []
    for inline, formset in zip(inline_instances, formsets):
        fieldsets = list(inline.get_fieldsets(request, obj))
        readonly = list(inline.get_readonly_fields(request, obj))
        if can_edit_parent:
            has_add_permission = inline.has_add_permission(request, obj)
            has_change_permission = inline.has_change_permission(request, obj)
            has_delete_permission = inline.has_delete_permission(request, obj)
        else:
            has_add_permission = has_change_permission = has_delete_permission = False
            formset.extra = formset.max_num = 0
        has_view_permission = inline.has_view_permission(request, obj)
        prepopulated = dict(inline.get_prepopulated_fields(request, obj))
        inline_admin_formset = helpers.InlineAdminFormSet(inline, formset, fieldsets, prepopulated, readonly, model_admin=self, has_add_permission=has_add_permission, has_change_permission=has_change_permission, has_delete_permission=has_delete_permission, has_view_permission=has_view_permission)
        inline_admin_formsets.append(inline_admin_formset)
    return inline_admin_formsets