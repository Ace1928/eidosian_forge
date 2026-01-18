import copy
import inspect
import warnings
from functools import partialmethod
from itertools import chain
from asgiref.sync import sync_to_async
import django
from django.apps import apps
from django.conf import settings
from django.core import checks
from django.core.exceptions import (
from django.db import (
from django.db.models import NOT_PROVIDED, ExpressionWrapper, IntegerField, Max, Value
from django.db.models.constants import LOOKUP_SEP
from django.db.models.constraints import CheckConstraint, UniqueConstraint
from django.db.models.deletion import CASCADE, Collector
from django.db.models.expressions import DatabaseDefault, RawSQL
from django.db.models.fields.related import (
from django.db.models.functions import Coalesce
from django.db.models.manager import Manager
from django.db.models.options import Options
from django.db.models.query import F, Q
from django.db.models.signals import (
from django.db.models.utils import AltersData, make_model_tuple
from django.utils.encoding import force_str
from django.utils.hashable import make_hashable
from django.utils.text import capfirst, get_text_list
from django.utils.translation import gettext_lazy as _
def date_error_message(self, lookup_type, field_name, unique_for):
    opts = self._meta
    field = opts.get_field(field_name)
    return ValidationError(message=field.error_messages['unique_for_date'], code='unique_for_date', params={'model': self, 'model_name': capfirst(opts.verbose_name), 'lookup_type': lookup_type, 'field': field_name, 'field_label': capfirst(field.verbose_name), 'date_field': unique_for, 'date_field_label': capfirst(opts.get_field(unique_for).verbose_name)})