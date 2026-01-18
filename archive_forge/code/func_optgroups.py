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
def optgroups(self, name, value, attr=None):
    """Return selected options based on the ModelChoiceIterator."""
    default = (None, [], 0)
    groups = [default]
    has_selected = False
    selected_choices = {str(v) for v in value if str(v) not in self.choices.field.empty_values}
    if not self.is_required and (not self.allow_multiple_selected):
        default[1].append(self.create_option(name, '', '', False, 0))
    remote_model_opts = self.field.remote_field.model._meta
    to_field_name = getattr(self.field.remote_field, 'field_name', remote_model_opts.pk.attname)
    to_field_name = remote_model_opts.get_field(to_field_name).attname
    choices = ((getattr(obj, to_field_name), self.choices.field.label_from_instance(obj)) for obj in self.choices.queryset.using(self.db).filter(**{'%s__in' % to_field_name: selected_choices}))
    for option_value, option_label in choices:
        selected = str(option_value) in value and (has_selected is False or self.allow_multiple_selected)
        has_selected |= selected
        index = len(default[1])
        subgroup = default[1]
        subgroup.append(self.create_option(name, option_value, option_label, selected_choices, index))
    return groups