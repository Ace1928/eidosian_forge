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
def process_rhs(self, compiler, connection):
    rhs, rhs_params = super().process_rhs(compiler, connection)
    cast_type = self.lhs.output_field.cast_db_type(connection)
    return ('%s::%s' % (rhs, cast_type), rhs_params)