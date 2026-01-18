from functools import partial
from django.contrib.admin.checks import InlineModelAdminChecks
from django.contrib.admin.options import InlineModelAdmin, flatten_fieldsets
from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.forms import (
from django.core import checks
from django.core.exceptions import FieldDoesNotExist
from django.forms import ALL_FIELDS
from django.forms.models import modelform_defines_fields
def get_formset(self, request, obj=None, **kwargs):
    if 'fields' in kwargs:
        fields = kwargs.pop('fields')
    else:
        fields = flatten_fieldsets(self.get_fieldsets(request, obj))
    exclude = [*(self.exclude or []), *self.get_readonly_fields(request, obj)]
    if self.exclude is None and hasattr(self.form, '_meta') and self.form._meta.exclude:
        exclude.extend(self.form._meta.exclude)
    exclude = exclude or None
    can_delete = self.can_delete and self.has_delete_permission(request, obj)
    defaults = {'ct_field': self.ct_field, 'fk_field': self.ct_fk_field, 'form': self.form, 'formfield_callback': partial(self.formfield_for_dbfield, request=request), 'formset': self.formset, 'extra': self.get_extra(request, obj), 'can_delete': can_delete, 'can_order': False, 'fields': fields, 'min_num': self.get_min_num(request, obj), 'max_num': self.get_max_num(request, obj), 'exclude': exclude, **kwargs}
    if defaults['fields'] is None and (not modelform_defines_fields(defaults['form'])):
        defaults['fields'] = ALL_FIELDS
    return generic_inlineformset_factory(self.model, **defaults)