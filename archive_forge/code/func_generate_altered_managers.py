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
def generate_altered_managers(self):
    for app_label, model_name in sorted(self.kept_model_keys):
        old_model_name = self.renamed_models.get((app_label, model_name), model_name)
        old_model_state = self.from_state.models[app_label, old_model_name]
        new_model_state = self.to_state.models[app_label, model_name]
        if old_model_state.managers != new_model_state.managers:
            self.add_operation(app_label, operations.AlterModelManagers(name=model_name, managers=new_model_state.managers))