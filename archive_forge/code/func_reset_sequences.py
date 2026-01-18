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
def reset_sequences(self, connection, models):
    """Reset database sequences for the given connection and models."""
    sequence_sql = connection.ops.sequence_reset_sql(no_style(), models)
    if sequence_sql:
        if self.verbosity >= 2:
            self.stdout.write('Resetting sequences')
        with connection.cursor() as cursor:
            for line in sequence_sql:
                cursor.execute(line)