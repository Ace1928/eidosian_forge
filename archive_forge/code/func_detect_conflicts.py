import pkgutil
import sys
from importlib import import_module, reload
from django.apps import apps
from django.conf import settings
from django.db.migrations.graph import MigrationGraph
from django.db.migrations.recorder import MigrationRecorder
from .exceptions import (
def detect_conflicts(self):
    """
        Look through the loaded graph and detect any conflicts - apps
        with more than one leaf migration. Return a dict of the app labels
        that conflict with the migration names that conflict.
        """
    seen_apps = {}
    conflicting_apps = set()
    for app_label, migration_name in self.graph.leaf_nodes():
        if app_label in seen_apps:
            conflicting_apps.add(app_label)
        seen_apps.setdefault(app_label, set()).add(migration_name)
    return {app_label: sorted(seen_apps[app_label]) for app_label in conflicting_apps}