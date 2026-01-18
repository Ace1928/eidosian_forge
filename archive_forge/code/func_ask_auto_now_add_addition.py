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
def ask_auto_now_add_addition(self, field_name, model_name):
    self.log_lack_of_migration(field_name, model_name, "it is impossible to add a field with 'auto_now_add=True' without specifying a default")
    sys.exit(3)