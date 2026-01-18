import pkgutil
import sys
from importlib import import_module, reload
from django.apps import apps
from django.conf import settings
from django.db.migrations.graph import MigrationGraph
from django.db.migrations.recorder import MigrationRecorder
from .exceptions import (
def get_migration(self, app_label, name_prefix):
    """Return the named migration or raise NodeNotFoundError."""
    return self.graph.nodes[app_label, name_prefix]