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
def loaddata(self, fixture_labels):
    connection = connections[self.using]
    self.fixture_count = 0
    self.loaded_object_count = 0
    self.fixture_object_count = 0
    self.models = set()
    self.serialization_formats = serializers.get_public_serializer_formats()
    for fixture_label in fixture_labels:
        if self.find_fixtures(fixture_label):
            break
    else:
        return
    self.objs_with_deferred_fields = []
    with connection.constraint_checks_disabled():
        for fixture_label in fixture_labels:
            self.load_label(fixture_label)
        for obj in self.objs_with_deferred_fields:
            obj.save_deferred_fields(using=self.using)
    table_names = [model._meta.db_table for model in self.models]
    try:
        connection.check_constraints(table_names=table_names)
    except Exception as e:
        e.args = ('Problem installing fixtures: %s' % e,)
        raise
    if self.loaded_object_count > 0:
        self.reset_sequences(connection, self.models)
    if self.verbosity >= 1:
        if self.fixture_object_count == self.loaded_object_count:
            self.stdout.write('Installed %d object(s) from %d fixture(s)' % (self.loaded_object_count, self.fixture_count))
        else:
            self.stdout.write('Installed %d object(s) (of %d) from %d fixture(s)' % (self.loaded_object_count, self.fixture_object_count, self.fixture_count))