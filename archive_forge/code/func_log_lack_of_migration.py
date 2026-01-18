import datetime
import importlib
import os
import sys
from django.apps import apps
from django.core.management.base import OutputWrapper
from django.db.models import NOT_PROVIDED
from django.utils import timezone
from django.utils.version import get_docs_version
from .loader import MigrationLoader
def log_lack_of_migration(self, field_name, model_name, reason):
    if self.verbosity > 0:
        self.log(f"Field '{field_name}' on model '{model_name}' not migrated: {reason}.")