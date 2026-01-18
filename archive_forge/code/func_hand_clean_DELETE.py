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
def hand_clean_DELETE(self):
    """
                We don't validate the 'DELETE' field itself because on
                templates it's not rendered using the field information, but
                just using a generic "deletion_field" of the InlineModelAdmin.
                """
    if self.cleaned_data.get(DELETION_FIELD_NAME, False):
        using = router.db_for_write(self._meta.model)
        collector = NestedObjects(using=using)
        if self.instance._state.adding:
            return
        collector.collect([self.instance])
        if collector.protected:
            objs = []
            for p in collector.protected:
                objs.append(_('%(class_name)s %(instance)s') % {'class_name': p._meta.verbose_name, 'instance': p})
            params = {'class_name': self._meta.model._meta.verbose_name, 'instance': self.instance, 'related_objects': get_text_list(objs, _('and'))}
            msg = _('Deleting %(class_name)s %(instance)s would require deleting the following protected related objects: %(related_objects)s')
            raise ValidationError(msg, code='deleting_protected', params=params)