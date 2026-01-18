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
def pk_is_not_editable(pk):
    return not pk.editable or (pk.auto_created or isinstance(pk, AutoField)) or (pk.remote_field and pk.remote_field.parent_link and pk_is_not_editable(pk.remote_field.model._meta.pk))