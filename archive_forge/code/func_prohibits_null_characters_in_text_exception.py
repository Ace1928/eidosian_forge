import operator
from django.db import DataError, InterfaceError
from django.db.backends.base.features import BaseDatabaseFeatures
from django.db.backends.postgresql.psycopg_any import is_psycopg3
from django.utils.functional import cached_property
@cached_property
def prohibits_null_characters_in_text_exception(self):
    if is_psycopg3:
        return (DataError, 'PostgreSQL text fields cannot contain NUL (0x00) bytes')
    else:
        return (ValueError, 'A string literal cannot contain NUL (0x00) characters.')