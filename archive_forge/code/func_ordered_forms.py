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
def ordered_forms(self):
    """
        Return a list of form in the order specified by the incoming data.
        Raise an AttributeError if ordering is not allowed.
        """
    if not self.is_valid() or not self.can_order:
        raise AttributeError("'%s' object has no attribute 'ordered_forms'" % self.__class__.__name__)
    if not hasattr(self, '_ordering'):
        self._ordering = []
        for i, form in enumerate(self.forms):
            if i >= self.initial_form_count() and (not form.has_changed()):
                continue
            if self.can_delete and self._should_delete_form(form):
                continue
            self._ordering.append((i, form.cleaned_data[ORDERING_FIELD_NAME]))

        def compare_ordering_key(k):
            if k[1] is None:
                return (1, 0)
            return (0, k[1])
        self._ordering.sort(key=compare_ordering_key)
    return [self.forms[i[0]] for i in self._ordering]