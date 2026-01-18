import keyword
import re
from django.core.management.base import BaseCommand, CommandError
from django.db import DEFAULT_DB_ALIAS, connections
from django.db.models.constants import LOOKUP_SEP
def normalize_table_name(self, table_name):
    """Translate the table name to a Python-compatible model name."""
    return re.sub('[^a-zA-Z0-9]', '', table_name.title())