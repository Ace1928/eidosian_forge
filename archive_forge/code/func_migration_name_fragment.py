from django.contrib.postgres.signals import (
from django.db import NotSupportedError, router
from django.db.migrations import AddConstraint, AddIndex, RemoveIndex
from django.db.migrations.operations.base import Operation
from django.db.models.constraints import CheckConstraint
@property
def migration_name_fragment(self):
    return '%s_validate_%s' % (self.model_name.lower(), self.name.lower())