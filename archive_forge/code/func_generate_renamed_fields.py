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
def generate_renamed_fields(self):
    """Generate RenameField operations."""
    for rem_app_label, rem_model_name, rem_db_column, rem_field_name, app_label, model_name, field, field_name in self.renamed_operations:
        if rem_db_column != field.db_column:
            altered_field = field.clone()
            altered_field.name = rem_field_name
            self.add_operation(app_label, operations.AlterField(model_name=model_name, name=rem_field_name, field=altered_field))
        self.add_operation(app_label, operations.RenameField(model_name=model_name, old_name=rem_field_name, new_name=field_name))
        self.old_field_keys.remove((rem_app_label, rem_model_name, rem_field_name))
        self.old_field_keys.add((app_label, model_name, field_name))