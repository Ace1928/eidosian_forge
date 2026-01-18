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
def render_multiple(self, model_states):
    if not model_states:
        return
    with self.bulk_update():
        unrendered_models = model_states
        while unrendered_models:
            new_unrendered_models = []
            for model in unrendered_models:
                try:
                    model.render(self)
                except InvalidBasesError:
                    new_unrendered_models.append(model)
            if len(new_unrendered_models) == len(unrendered_models):
                raise InvalidBasesError('Cannot resolve bases for %r\nThis can happen if you are inheriting models from an app with migrations (e.g. contrib.auth)\n in an app with no migrations; see https://docs.djangoproject.com/en/%s/topics/migrations/#dependencies for more' % (new_unrendered_models, get_docs_version()))
            unrendered_models = new_unrendered_models