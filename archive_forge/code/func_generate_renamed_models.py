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
def generate_renamed_models(self):
    """
        Find any renamed models, generate the operations for them, and remove
        the old entry from the model lists. Must be run before other
        model-level generation.
        """
    self.renamed_models = {}
    self.renamed_models_rel = {}
    added_models = self.new_model_keys - self.old_model_keys
    for app_label, model_name in sorted(added_models):
        model_state = self.to_state.models[app_label, model_name]
        model_fields_def = self.only_relation_agnostic_fields(model_state.fields)
        removed_models = self.old_model_keys - self.new_model_keys
        for rem_app_label, rem_model_name in removed_models:
            if rem_app_label == app_label:
                rem_model_state = self.from_state.models[rem_app_label, rem_model_name]
                rem_model_fields_def = self.only_relation_agnostic_fields(rem_model_state.fields)
                if model_fields_def == rem_model_fields_def:
                    if self.questioner.ask_rename_model(rem_model_state, model_state):
                        dependencies = []
                        fields = list(model_state.fields.values()) + [field.remote_field for relations in self.to_state.relations[app_label, model_name].values() for field in relations.values()]
                        for field in fields:
                            if field.is_relation:
                                dependencies.extend(self._get_dependencies_for_foreign_key(app_label, model_name, field, self.to_state))
                        self.add_operation(app_label, operations.RenameModel(old_name=rem_model_state.name, new_name=model_state.name), dependencies=dependencies)
                        self.renamed_models[app_label, model_name] = rem_model_name
                        renamed_models_rel_key = '%s.%s' % (rem_model_state.app_label, rem_model_state.name_lower)
                        self.renamed_models_rel[renamed_models_rel_key] = '%s.%s' % (model_state.app_label, model_state.name_lower)
                        self.old_model_keys.remove((rem_app_label, rem_model_name))
                        self.old_model_keys.add((app_label, model_name))
                        break