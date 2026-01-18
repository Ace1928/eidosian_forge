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
def rename_model(self, app_label, old_name, new_name):
    old_name_lower = old_name.lower()
    new_name_lower = new_name.lower()
    renamed_model = self.models[app_label, old_name_lower].clone()
    renamed_model.name = new_name
    self.models[app_label, new_name_lower] = renamed_model
    old_model_tuple = (app_label, old_name_lower)
    new_remote_model = f'{app_label}.{new_name}'
    to_reload = set()
    for model_state, name, field, reference in get_references(self, old_model_tuple):
        changed_field = None
        if reference.to:
            changed_field = field.clone()
            changed_field.remote_field.model = new_remote_model
        if reference.through:
            if changed_field is None:
                changed_field = field.clone()
            changed_field.remote_field.through = new_remote_model
        if changed_field:
            model_state.fields[name] = changed_field
            to_reload.add((model_state.app_label, model_state.name_lower))
    if self._relations is not None:
        old_name_key = (app_label, old_name_lower)
        new_name_key = (app_label, new_name_lower)
        if old_name_key in self._relations:
            self._relations[new_name_key] = self._relations.pop(old_name_key)
        for model_relations in self._relations.values():
            if old_name_key in model_relations:
                model_relations[new_name_key] = model_relations.pop(old_name_key)
    self.reload_models(to_reload, delay=True)
    self.remove_model(app_label, old_name_lower)
    self.reload_model(app_label, new_name_lower, delay=True)