import copy
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from django.apps import AppConfig
from django.apps.registry import Apps
from django.apps.registry import apps as global_apps
from django.conf import settings
from django.core.exceptions import FieldDoesNotExist
from django.db import models
from django.db.migrations.utils import field_is_referenced, get_references
from django.db.models import NOT_PROVIDED
from django.db.models.fields.related import RECURSIVE_RELATIONSHIP_CONSTANT
from django.db.models.options import DEFAULT_NAMES, normalize_together
from django.db.models.utils import make_model_tuple
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
from django.utils.version import get_docs_version
from .exceptions import InvalidBasesError
from .utils import resolve_relation
def construct_managers(self):
    """Deep-clone the managers using deconstruction."""
    sorted_managers = sorted(self.managers, key=lambda v: v[1].creation_counter)
    for mgr_name, manager in sorted_managers:
        as_manager, manager_path, qs_path, args, kwargs = manager.deconstruct()
        if as_manager:
            qs_class = import_string(qs_path)
            yield (mgr_name, qs_class.as_manager())
        else:
            manager_class = import_string(manager_path)
            yield (mgr_name, manager_class(*args, **kwargs))