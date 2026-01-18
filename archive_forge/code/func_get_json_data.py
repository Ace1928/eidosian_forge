import json
from collections import UserList
from django.conf import settings
from django.core.exceptions import ValidationError
from django.forms.renderers import get_default_renderer
from django.utils import timezone
from django.utils.html import escape, format_html_join
from django.utils.safestring import mark_safe
from django.utils.translation import gettext_lazy as _
def get_json_data(self, escape_html=False):
    errors = []
    for error in self.as_data():
        message = next(iter(error))
        errors.append({'message': escape(message) if escape_html else message, 'code': error.code or ''})
    return errors