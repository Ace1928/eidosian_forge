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
def get_view_on_site_url(self, obj=None):
    if obj is None or not self.view_on_site:
        return None
    if callable(self.view_on_site):
        return self.view_on_site(obj)
    elif hasattr(obj, 'get_absolute_url'):
        return reverse('admin:view_on_site', kwargs={'content_type_id': get_content_type_for_model(obj).pk, 'object_id': obj.pk}, current_app=self.admin_site.name)