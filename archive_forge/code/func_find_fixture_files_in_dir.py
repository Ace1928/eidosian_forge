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
def find_fixture_files_in_dir(self, fixture_dir, fixture_name, targets):
    fixture_files_in_dir = []
    path = os.path.join(fixture_dir, fixture_name)
    for candidate in glob.iglob(glob.escape(path) + '*'):
        if os.path.basename(candidate) in targets:
            fixture_files_in_dir.append((candidate, fixture_dir, fixture_name))
    return fixture_files_in_dir