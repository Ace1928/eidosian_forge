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
def get_forward_related_filter(self, obj):
    """See corresponding method on RelatedField"""
    return {self.fk_field: obj.pk, self.ct_field: ContentType.objects.get_for_model(obj).pk}