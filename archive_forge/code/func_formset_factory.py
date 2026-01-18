from django.core.exceptions import ValidationError
from django.forms import Form
from django.forms.fields import BooleanField, IntegerField
from django.forms.renderers import get_default_renderer
from django.forms.utils import ErrorList, RenderableFormMixin
from django.forms.widgets import CheckboxInput, HiddenInput, NumberInput
from django.utils.functional import cached_property
from django.utils.translation import gettext_lazy as _
from django.utils.translation import ngettext_lazy
def formset_factory(form, formset=BaseFormSet, extra=1, can_order=False, can_delete=False, max_num=None, validate_max=False, min_num=None, validate_min=False, absolute_max=None, can_delete_extra=True, renderer=None):
    """Return a FormSet for the given form class."""
    if min_num is None:
        min_num = DEFAULT_MIN_NUM
    if max_num is None:
        max_num = DEFAULT_MAX_NUM
    if absolute_max is None:
        absolute_max = max_num + DEFAULT_MAX_NUM
    if max_num > absolute_max:
        raise ValueError("'absolute_max' must be greater or equal to 'max_num'.")
    attrs = {'form': form, 'extra': extra, 'can_order': can_order, 'can_delete': can_delete, 'can_delete_extra': can_delete_extra, 'min_num': min_num, 'max_num': max_num, 'absolute_max': absolute_max, 'validate_min': validate_min, 'validate_max': validate_max, 'renderer': renderer}
    return type(form.__name__ + 'FormSet', (formset,), attrs)