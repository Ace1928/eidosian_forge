import functools
import re
from collections import defaultdict
from graphlib import TopologicalSorter
from itertools import chain
from django.conf import settings
from django.db import models
from django.db.migrations import operations
from django.db.migrations.migration import Migration
from django.db.migrations.operations.models import AlterModelOptions
from django.db.migrations.optimizer import MigrationOptimizer
from django.db.migrations.questioner import MigrationQuestioner
from django.db.migrations.utils import (
def only_relation_agnostic_fields(self, fields):
    """
        Return a definition of the fields that ignores field names and
        what related fields actually relate to. Used for detecting renames (as
        the related fields change during renames).
        """
    fields_def = []
    for name, field in sorted(fields.items()):
        deconstruction = self.deep_deconstruct(field)
        if field.remote_field and field.remote_field.model:
            deconstruction[2].pop('to', None)
        fields_def.append(deconstruction)
    return fields_def