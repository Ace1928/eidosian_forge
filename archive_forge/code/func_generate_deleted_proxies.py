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
def generate_deleted_proxies(self):
    """Make DeleteModel options for proxy models."""
    deleted = self.old_proxy_keys - self.new_proxy_keys
    for app_label, model_name in sorted(deleted):
        model_state = self.from_state.models[app_label, model_name]
        assert model_state.options.get('proxy')
        self.add_operation(app_label, operations.DeleteModel(name=model_state.name))