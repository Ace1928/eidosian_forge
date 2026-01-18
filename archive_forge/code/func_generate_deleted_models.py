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
def generate_deleted_models(self):
    """
        Find all deleted models (managed and unmanaged) and make delete
        operations for them as well as separate operations to delete any
        foreign key or M2M relationships (these are optimized later, if
        possible).

        Also bring forward removal of any model options that refer to
        collections of fields - the inverse of generate_created_models().
        """
    new_keys = self.new_model_keys | self.new_unmanaged_keys
    deleted_models = self.old_model_keys - new_keys
    deleted_unmanaged_models = self.old_unmanaged_keys - new_keys
    all_deleted_models = chain(sorted(deleted_models), sorted(deleted_unmanaged_models))
    for app_label, model_name in all_deleted_models:
        model_state = self.from_state.models[app_label, model_name]
        related_fields = {}
        for field_name, field in model_state.fields.items():
            if field.remote_field:
                if field.remote_field.model:
                    related_fields[field_name] = field
                if getattr(field.remote_field, 'through', None):
                    related_fields[field_name] = field
        unique_together = model_state.options.pop('unique_together', None)
        index_together = model_state.options.pop('index_together', None)
        if unique_together:
            self.add_operation(app_label, operations.AlterUniqueTogether(name=model_name, unique_together=None))
        if index_together:
            self.add_operation(app_label, operations.AlterIndexTogether(name=model_name, index_together=None))
        for name in sorted(related_fields):
            self.add_operation(app_label, operations.RemoveField(model_name=model_name, name=name))
        dependencies = []
        relations = self.from_state.relations
        for (related_object_app_label, object_name), relation_related_fields in relations[app_label, model_name].items():
            for field_name, field in relation_related_fields.items():
                dependencies.append((related_object_app_label, object_name, field_name, False))
                if not field.many_to_many:
                    dependencies.append((related_object_app_label, object_name, field_name, 'alter'))
        for name in sorted(related_fields):
            dependencies.append((app_label, model_name, name, False))
        through_user = self.through_users.get((app_label, model_state.name_lower))
        if through_user:
            dependencies.append((through_user[0], through_user[1], through_user[2], False))
        self.add_operation(app_label, operations.DeleteModel(name=model_state.name), dependencies=list(set(dependencies)))