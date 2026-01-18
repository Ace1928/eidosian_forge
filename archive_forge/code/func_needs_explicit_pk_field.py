import json
from django import forms
from django.contrib.admin.utils import (
from django.core.exceptions import ObjectDoesNotExist
from django.db.models.fields.related import (
from django.forms.utils import flatatt
from django.template.defaultfilters import capfirst, linebreaksbr
from django.urls import NoReverseMatch, reverse
from django.utils.html import conditional_escape, format_html
from django.utils.safestring import mark_safe
from django.utils.translation import gettext
from django.utils.translation import gettext_lazy as _
def needs_explicit_pk_field(self):
    return self.form._meta.model._meta.auto_field or not self.form._meta.model._meta.pk.editable or any((parent._meta.auto_field or not parent._meta.model._meta.pk.editable for parent in self.form._meta.model._meta.get_parent_list()))