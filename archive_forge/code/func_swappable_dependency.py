import re
from django.db.migrations.utils import get_migration_name_timestamp
from django.db.transaction import atomic
from .exceptions import IrreversibleError
def swappable_dependency(value):
    """Turn a setting value into a dependency."""
    return SwappableTuple((value.split('.', 1)[0], '__first__'), value)