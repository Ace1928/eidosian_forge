from itertools import chain
from django.core.exceptions import (
from django.db.models.utils import AltersData
from django.forms.fields import ChoiceField, Field
from django.forms.forms import BaseForm, DeclarativeFieldsMetaclass
from django.forms.formsets import BaseFormSet, formset_factory
from django.forms.utils import ErrorList
from django.forms.widgets import (
from django.utils.choices import BaseChoiceIterator
from django.utils.text import capfirst, get_text_list
from django.utils.translation import gettext
from django.utils.translation import gettext_lazy as _
def save_new_objects(self, commit=True):
    self.new_objects = []
    for form in self.extra_forms:
        if not form.has_changed():
            continue
        if self.can_delete and self._should_delete_form(form):
            continue
        self.new_objects.append(self.save_new(form, commit=commit))
        if not commit:
            self.saved_forms.append(form)
    return self.new_objects