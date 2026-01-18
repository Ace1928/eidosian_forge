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
def save_existing_objects(self, commit=True):
    self.changed_objects = []
    self.deleted_objects = []
    if not self.initial_forms:
        return []
    saved_instances = []
    forms_to_delete = self.deleted_forms
    for form in self.initial_forms:
        obj = form.instance
        if obj.pk is None:
            continue
        if form in forms_to_delete:
            self.deleted_objects.append(obj)
            self.delete_existing(obj, commit=commit)
        elif form.has_changed():
            self.changed_objects.append((obj, form.changed_data))
            saved_instances.append(self.save_existing(form, obj, commit=commit))
            if not commit:
                self.saved_forms.append(form)
    return saved_instances