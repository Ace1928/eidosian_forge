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
def ask_unique_callable_default_addition(self, field_name, model_name):
    """Adding a unique field with a callable default."""
    if not self.dry_run:
        version = get_docs_version()
        choice = self._choice_input(f'Callable default on unique field {model_name}.{field_name} will not generate unique values upon migrating.\nPlease choose how to proceed:\n', [f'Continue making this migration as the first step in writing a manual migration to generate unique values described here: https://docs.djangoproject.com/en/{version}/howto/writing-migrations/#migrations-that-add-unique-fields.', 'Quit and edit field options in models.py.'])
        if choice == 2:
            sys.exit(3)
    return None