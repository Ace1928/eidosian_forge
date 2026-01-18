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
def deleted_forms(self):
    """Return a list of forms that have been marked for deletion."""
    if not self.is_valid() or not self.can_delete:
        return []
    if not hasattr(self, '_deleted_form_indexes'):
        self._deleted_form_indexes = []
        for i, form in enumerate(self.forms):
            if i >= self.initial_form_count() and (not form.has_changed()):
                continue
            if self._should_delete_form(form):
                self._deleted_form_indexes.append(i)
    return [self.forms[i] for i in self._deleted_form_indexes]