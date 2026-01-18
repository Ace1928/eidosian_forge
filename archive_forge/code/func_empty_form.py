from django.core.exceptions import ValidationError
from django.forms import Form
from django.forms.fields import BooleanField, IntegerField
from django.forms.renderers import get_default_renderer
from django.forms.utils import ErrorList, RenderableFormMixin
from django.forms.widgets import CheckboxInput, HiddenInput, NumberInput
from django.utils.functional import cached_property
from django.utils.translation import gettext_lazy as _
from django.utils.translation import ngettext_lazy
@property
def empty_form(self):
    form_kwargs = {**self.get_form_kwargs(None), 'auto_id': self.auto_id, 'prefix': self.add_prefix('__prefix__'), 'empty_permitted': True, 'use_required_attribute': False, 'renderer': self.form_renderer}
    form = self.form(**form_kwargs)
    self.add_fields(form, None)
    return form