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
def base_url_parameters(self):
    limit_choices_to = self.rel.limit_choices_to
    if callable(limit_choices_to):
        limit_choices_to = limit_choices_to()
    return url_params_from_lookup_dict(limit_choices_to)