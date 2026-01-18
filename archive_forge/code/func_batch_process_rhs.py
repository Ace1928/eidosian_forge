import itertools
import math
import warnings
from django.core.exceptions import EmptyResultSet, FullResultSet
from django.db.backends.base.operations import BaseDatabaseOperations
from django.db.models.expressions import Case, Expression, Func, Value, When
from django.db.models.fields import (
from django.db.models.query_utils import RegisterLookupMixin
from django.utils.datastructures import OrderedSet
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.functional import cached_property
from django.utils.hashable import make_hashable
def batch_process_rhs(self, compiler, connection, rhs=None):
    pre_processed = super().batch_process_rhs(compiler, connection, rhs)
    sql, params = zip(*(self.resolve_expression_parameter(compiler, connection, sql, param) for sql, param in zip(*pre_processed)))
    params = itertools.chain.from_iterable(params)
    return (sql, tuple(params))