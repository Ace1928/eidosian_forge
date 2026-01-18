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
def generate_added_constraints(self):
    for (app_label, model_name), alt_constraints in self.altered_constraints.items():
        dependencies = self._get_dependencies_for_model(app_label, model_name)
        for constraint in alt_constraints['added_constraints']:
            self.add_operation(app_label, operations.AddConstraint(model_name=model_name, constraint=constraint), dependencies=dependencies)