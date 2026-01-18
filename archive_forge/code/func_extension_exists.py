from django.contrib.postgres.signals import (
from django.db import NotSupportedError, router
from django.db.migrations import AddConstraint, AddIndex, RemoveIndex
from django.db.migrations.operations.base import Operation
from django.db.models.constraints import CheckConstraint
def extension_exists(self, schema_editor, extension):
    with schema_editor.connection.cursor() as cursor:
        cursor.execute('SELECT 1 FROM pg_extension WHERE extname = %s', [extension])
        return bool(cursor.fetchone())