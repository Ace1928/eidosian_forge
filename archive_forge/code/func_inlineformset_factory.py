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
def inlineformset_factory(parent_model, model, form=ModelForm, formset=BaseInlineFormSet, fk_name=None, fields=None, exclude=None, extra=3, can_order=False, can_delete=True, max_num=None, formfield_callback=None, widgets=None, validate_max=False, localized_fields=None, labels=None, help_texts=None, error_messages=None, min_num=None, validate_min=False, field_classes=None, absolute_max=None, can_delete_extra=True, renderer=None, edit_only=False):
    """
    Return an ``InlineFormSet`` for the given kwargs.

    ``fk_name`` must be provided if ``model`` has more than one ``ForeignKey``
    to ``parent_model``.
    """
    fk = _get_foreign_key(parent_model, model, fk_name=fk_name)
    if fk.unique:
        max_num = 1
    kwargs = {'form': form, 'formfield_callback': formfield_callback, 'formset': formset, 'extra': extra, 'can_delete': can_delete, 'can_order': can_order, 'fields': fields, 'exclude': exclude, 'min_num': min_num, 'max_num': max_num, 'widgets': widgets, 'validate_min': validate_min, 'validate_max': validate_max, 'localized_fields': localized_fields, 'labels': labels, 'help_texts': help_texts, 'error_messages': error_messages, 'field_classes': field_classes, 'absolute_max': absolute_max, 'can_delete_extra': can_delete_extra, 'renderer': renderer, 'edit_only': edit_only}
    FormSet = modelformset_factory(model, **kwargs)
    FormSet.fk = fk
    return FormSet