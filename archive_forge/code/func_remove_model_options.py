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
def remove_model_options(self, app_label, model_name, option_name, value_to_remove):
    model_state = self.models[app_label, model_name]
    if (objs := model_state.options.get(option_name)):
        model_state.options[option_name] = [obj for obj in objs if tuple(obj) != tuple(value_to_remove)]
    self.reload_model(app_label, model_name, delay=True)