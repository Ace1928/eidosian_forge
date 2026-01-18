from django.core.exceptions import ValidationError
from django.forms import Form
from django.forms.fields import BooleanField, IntegerField
from django.forms.renderers import get_default_renderer
from django.forms.utils import ErrorList, RenderableFormMixin
from django.forms.widgets import CheckboxInput, HiddenInput, NumberInput
from django.utils.functional import cached_property
from django.utils.translation import gettext_lazy as _
from django.utils.translation import ngettext_lazy
@cached_property
def management_form(self):
    """Return the ManagementForm instance for this FormSet."""
    if self.is_bound:
        form = ManagementForm(self.data, auto_id=self.auto_id, prefix=self.prefix, renderer=self.renderer)
        form.full_clean()
    else:
        form = ManagementForm(auto_id=self.auto_id, prefix=self.prefix, initial={TOTAL_FORM_COUNT: self.total_form_count(), INITIAL_FORM_COUNT: self.initial_form_count(), MIN_NUM_FORM_COUNT: self.min_num, MAX_NUM_FORM_COUNT: self.max_num}, renderer=self.renderer)
    return form