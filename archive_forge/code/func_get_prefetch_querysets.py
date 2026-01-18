import functools
import itertools
import warnings
from collections import defaultdict
from asgiref.sync import sync_to_async
from django.contrib.contenttypes.models import ContentType
from django.core import checks
from django.core.exceptions import FieldDoesNotExist, ObjectDoesNotExist
from django.db import DEFAULT_DB_ALIAS, models, router, transaction
from django.db.models import DO_NOTHING, ForeignObject, ForeignObjectRel
from django.db.models.base import ModelBase, make_foreign_order_accessors
from django.db.models.fields.mixins import FieldCacheMixin
from django.db.models.fields.related import (
from django.db.models.query_utils import PathInfo
from django.db.models.sql import AND
from django.db.models.sql.where import WhereNode
from django.db.models.utils import AltersData
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.functional import cached_property
def get_prefetch_querysets(self, instances, querysets=None):
    if querysets and len(querysets) != 1:
        raise ValueError('querysets argument of get_prefetch_querysets() should have a length of 1.')
    queryset = querysets[0] if querysets else super().get_queryset()
    queryset._add_hints(instance=instances[0])
    queryset = queryset.using(queryset._db or self._db)
    content_type_queries = [models.Q.create([(f'{self.content_type_field_name}__pk', content_type_id), (f'{self.object_id_field_name}__in', {obj.pk for obj in objs})]) for content_type_id, objs in itertools.groupby(sorted(instances, key=lambda obj: self.get_content_type(obj).pk), lambda obj: self.get_content_type(obj).pk)]
    query = models.Q.create(content_type_queries, connector=models.Q.OR)
    object_id_converter = instances[0]._meta.pk.to_python
    content_type_id_field_name = '%s_id' % self.content_type_field_name
    return (queryset.filter(query), lambda relobj: (object_id_converter(getattr(relobj, self.object_id_field_name)), getattr(relobj, content_type_id_field_name)), lambda obj: (obj.pk, self.get_content_type(obj).pk), False, self.prefetch_cache_name, False)