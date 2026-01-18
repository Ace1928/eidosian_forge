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
def full_clean(self, exclude=None, validate_unique=True, validate_constraints=True):
    """
        Call clean_fields(), clean(), validate_unique(), and
        validate_constraints() on the model. Raise a ValidationError for any
        errors that occur.
        """
    errors = {}
    if exclude is None:
        exclude = set()
    else:
        exclude = set(exclude)
    try:
        self.clean_fields(exclude=exclude)
    except ValidationError as e:
        errors = e.update_error_dict(errors)
    try:
        self.clean()
    except ValidationError as e:
        errors = e.update_error_dict(errors)
    if validate_unique:
        for name in errors:
            if name != NON_FIELD_ERRORS and name not in exclude:
                exclude.add(name)
        try:
            self.validate_unique(exclude=exclude)
        except ValidationError as e:
            errors = e.update_error_dict(errors)
    if validate_constraints:
        for name in errors:
            if name != NON_FIELD_ERRORS and name not in exclude:
                exclude.add(name)
        try:
            self.validate_constraints(exclude=exclude)
        except ValidationError as e:
            errors = e.update_error_dict(errors)
    if errors:
        raise ValidationError(errors)