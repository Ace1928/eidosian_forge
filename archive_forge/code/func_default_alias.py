from django.core.exceptions import FieldError, FullResultSet
from django.db.models.expressions import Case, Func, Star, Value, When
from django.db.models.fields import IntegerField
from django.db.models.functions.comparison import Coalesce
from django.db.models.functions.mixins import (
@property
def default_alias(self):
    expressions = self.get_source_expressions()
    if len(expressions) == 1 and hasattr(expressions[0], 'name'):
        return '%s__%s' % (expressions[0].name, self.name.lower())
    raise TypeError('Complex expressions require an alias')