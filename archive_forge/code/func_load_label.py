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
def load_label(self, fixture_label):
    """Load fixtures files for a given label."""
    show_progress = self.verbosity >= 3
    for fixture_file, fixture_dir, fixture_name in self.find_fixtures(fixture_label):
        _, ser_fmt, cmp_fmt = self.parse_name(os.path.basename(fixture_file))
        open_method, mode = self.compression_formats[cmp_fmt]
        fixture = open_method(fixture_file, mode)
        self.fixture_count += 1
        objects_in_fixture = 0
        loaded_objects_in_fixture = 0
        if self.verbosity >= 2:
            self.stdout.write("Installing %s fixture '%s' from %s." % (ser_fmt, fixture_name, humanize(fixture_dir)))
        try:
            objects = serializers.deserialize(ser_fmt, fixture, using=self.using, ignorenonexistent=self.ignore, handle_forward_references=True)
            for obj in objects:
                objects_in_fixture += 1
                if self.save_obj(obj):
                    loaded_objects_in_fixture += 1
                    if show_progress:
                        self.stdout.write('\rProcessed %i object(s).' % loaded_objects_in_fixture, ending='')
        except Exception as e:
            if not isinstance(e, CommandError):
                e.args = ("Problem installing fixture '%s': %s" % (fixture_file, e),)
            raise
        finally:
            fixture.close()
        if objects_in_fixture and show_progress:
            self.stdout.write()
        self.loaded_object_count += loaded_objects_in_fixture
        self.fixture_object_count += objects_in_fixture
        if objects_in_fixture == 0:
            warnings.warn("No fixture data found for '%s'. (File format may be invalid.)" % fixture_name, RuntimeWarning)