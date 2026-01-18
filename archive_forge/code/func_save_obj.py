import functools
import glob
import gzip
import os
import sys
import warnings
import zipfile
from itertools import product
from django.apps import apps
from django.conf import settings
from django.core import serializers
from django.core.exceptions import ImproperlyConfigured
from django.core.management.base import BaseCommand, CommandError
from django.core.management.color import no_style
from django.core.management.utils import parse_apps_and_model_labels
from django.db import (
from django.utils.functional import cached_property
def save_obj(self, obj):
    """Save an object if permitted."""
    if obj.object._meta.app_config in self.excluded_apps or type(obj.object) in self.excluded_models:
        return False
    saved = False
    if router.allow_migrate_model(self.using, obj.object.__class__):
        saved = True
        self.models.add(obj.object.__class__)
        try:
            obj.save(using=self.using)
        except (DatabaseError, IntegrityError, ValueError) as e:
            e.args = ('Could not load %(object_label)s(pk=%(pk)s): %(error_msg)s' % {'object_label': obj.object._meta.label, 'pk': obj.object.pk, 'error_msg': e},)
            raise
    if obj.deferred_fields:
        self.objs_with_deferred_fields.append(obj)
    return saved