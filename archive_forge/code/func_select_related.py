import copy
import operator
import warnings
from itertools import chain, islice
from asgiref.sync import sync_to_async
import django
from django.conf import settings
from django.core import exceptions
from django.db import (
from django.db.models import AutoField, DateField, DateTimeField, Field, sql
from django.db.models.constants import LOOKUP_SEP, OnConflict
from django.db.models.deletion import Collector
from django.db.models.expressions import Case, F, Value, When
from django.db.models.functions import Cast, Trunc
from django.db.models.query_utils import FilteredRelation, Q
from django.db.models.sql.constants import CURSOR, GET_ITERATOR_CHUNK_SIZE
from django.db.models.utils import (
from django.utils import timezone
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.functional import cached_property, partition
def select_related(self, *fields):
    """
        Return a new QuerySet instance that will select related objects.

        If fields are specified, they must be ForeignKey fields and only those
        related objects are included in the selection.

        If select_related(None) is called, clear the list.
        """
    self._not_support_combined_queries('select_related')
    if self._fields is not None:
        raise TypeError('Cannot call select_related() after .values() or .values_list()')
    obj = self._chain()
    if fields == (None,):
        obj.query.select_related = False
    elif fields:
        obj.query.add_select_related(fields)
    else:
        obj.query.select_related = True
    return obj