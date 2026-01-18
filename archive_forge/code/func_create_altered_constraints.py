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
def create_altered_constraints(self):
    option_name = operations.AddConstraint.option_name
    for app_label, model_name in sorted(self.kept_model_keys):
        old_model_name = self.renamed_models.get((app_label, model_name), model_name)
        old_model_state = self.from_state.models[app_label, old_model_name]
        new_model_state = self.to_state.models[app_label, model_name]
        old_constraints = old_model_state.options[option_name]
        new_constraints = new_model_state.options[option_name]
        add_constraints = [c for c in new_constraints if c not in old_constraints]
        rem_constraints = [c for c in old_constraints if c not in new_constraints]
        self.altered_constraints.update({(app_label, model_name): {'added_constraints': add_constraints, 'removed_constraints': rem_constraints}})