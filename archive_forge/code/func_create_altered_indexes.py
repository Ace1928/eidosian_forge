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
def create_altered_indexes(self):
    option_name = operations.AddIndex.option_name
    self.renamed_index_together_values = defaultdict(list)
    for app_label, model_name in sorted(self.kept_model_keys):
        old_model_name = self.renamed_models.get((app_label, model_name), model_name)
        old_model_state = self.from_state.models[app_label, old_model_name]
        new_model_state = self.to_state.models[app_label, model_name]
        old_indexes = old_model_state.options[option_name]
        new_indexes = new_model_state.options[option_name]
        added_indexes = [idx for idx in new_indexes if idx not in old_indexes]
        removed_indexes = [idx for idx in old_indexes if idx not in new_indexes]
        renamed_indexes = []
        remove_from_added = []
        remove_from_removed = []
        for new_index in added_indexes:
            new_index_dec = new_index.deconstruct()
            new_index_name = new_index_dec[2].pop('name')
            for old_index in removed_indexes:
                old_index_dec = old_index.deconstruct()
                old_index_name = old_index_dec[2].pop('name')
                if new_index_dec == old_index_dec and new_index_name != old_index_name:
                    renamed_indexes.append((old_index_name, new_index_name, None))
                    remove_from_added.append(new_index)
                    remove_from_removed.append(old_index)
        for old_value, new_value, index_together_app_label, index_together_model_name, dependencies in self._get_altered_foo_together_operations(operations.AlterIndexTogether.option_name):
            if app_label != index_together_app_label or model_name != index_together_model_name:
                continue
            removed_values = old_value.difference(new_value)
            for removed_index_together in removed_values:
                renamed_index_together_indexes = []
                for new_index in added_indexes:
                    _, args, kwargs = new_index.deconstruct()
                    if not args and new_index.fields == list(removed_index_together) and (set(kwargs) == {'name', 'fields'}):
                        renamed_index_together_indexes.append(new_index)
                if len(renamed_index_together_indexes) == 1:
                    renamed_index = renamed_index_together_indexes[0]
                    remove_from_added.append(renamed_index)
                    renamed_indexes.append((None, renamed_index.name, removed_index_together))
                    self.renamed_index_together_values[index_together_app_label, index_together_model_name].append(removed_index_together)
        added_indexes = [idx for idx in added_indexes if idx not in remove_from_added]
        removed_indexes = [idx for idx in removed_indexes if idx not in remove_from_removed]
        self.altered_indexes.update({(app_label, model_name): {'added_indexes': added_indexes, 'removed_indexes': removed_indexes, 'renamed_indexes': renamed_indexes}})