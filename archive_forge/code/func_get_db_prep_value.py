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
def get_db_prep_value(self, value, connection, prepared=False):
    if isinstance(value, (list, tuple)):
        return [self.base_field.get_db_prep_value(i, connection, prepared=False) for i in value]
    return value