import copy
import json
from django import forms
from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.validators import URLValidator
from django.db.models import CASCADE, UUIDField
from django.urls import reverse
from django.urls.exceptions import NoReverseMatch
from django.utils.html import smart_urlquote
from django.utils.http import urlencode
from django.utils.text import Truncator
from django.utils.translation import get_language
from django.utils.translation import gettext as _
def build_attrs(self, base_attrs, extra_attrs=None):
    """
        Set select2's AJAX attributes.

        Attributes can be set using the html5 data attribute.
        Nested attributes require a double dash as per
        https://select2.org/configuration/data-attributes#nested-subkey-options
        """
    attrs = super().build_attrs(base_attrs, extra_attrs=extra_attrs)
    attrs.setdefault('class', '')
    attrs.update({'data-ajax--cache': 'true', 'data-ajax--delay': 250, 'data-ajax--type': 'GET', 'data-ajax--url': self.get_url(), 'data-app-label': self.field.model._meta.app_label, 'data-model-name': self.field.model._meta.model_name, 'data-field-name': self.field.name, 'data-theme': 'admin-autocomplete', 'data-allow-clear': json.dumps(not self.is_required), 'data-placeholder': '', 'lang': self.i18n_name, 'class': attrs['class'] + (' ' if attrs['class'] else '') + 'admin-autocomplete'})
    return attrs