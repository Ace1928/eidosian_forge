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
def add_operation(self, app_label, operation, dependencies=None, beginning=False):
    operation._auto_deps = dependencies or []
    if beginning:
        self.generated_operations.setdefault(app_label, []).insert(0, operation)
    else:
        self.generated_operations.setdefault(app_label, []).append(operation)