import json
from django.contrib.postgres import lookups
from django.contrib.postgres.forms import SimpleArrayField
from django.contrib.postgres.validators import ArrayMaxLengthValidator
from django.core import checks, exceptions
from django.db.models import Field, Func, IntegerField, Transform, Value
from django.db.models.fields.mixins import CheckFieldDefaultMixin
from django.db.models.lookups import Exact, In
from django.utils.translation import gettext_lazy as _
from ..utils import prefix_validation_error
from .utils import AttributeSetter
def run_validators(self, value):
    super().run_validators(value)
    for index, part in enumerate(value):
        try:
            self.base_field.run_validators(part)
        except exceptions.ValidationError as error:
            raise prefix_validation_error(error, prefix=self.error_messages['item_invalid'], code='item_invalid', params={'nth': index + 1})