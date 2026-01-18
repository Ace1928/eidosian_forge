import keyword
import re
from django.core.management.base import BaseCommand, CommandError
from django.db import DEFAULT_DB_ALIAS, connections
from django.db.models.constants import LOOKUP_SEP
def normalize_col_name(self, col_name, used_column_names, is_relation):
    """
        Modify the column name to make it Python-compatible as a field name
        """
    field_params = {}
    field_notes = []
    new_name = col_name.lower()
    if new_name != col_name:
        field_notes.append('Field name made lowercase.')
    if is_relation:
        if new_name.endswith('_id'):
            new_name = new_name.removesuffix('_id')
        else:
            field_params['db_column'] = col_name
    new_name, num_repl = re.subn('\\W', '_', new_name)
    if num_repl > 0:
        field_notes.append('Field renamed to remove unsuitable characters.')
    if new_name.find(LOOKUP_SEP) >= 0:
        while new_name.find(LOOKUP_SEP) >= 0:
            new_name = new_name.replace(LOOKUP_SEP, '_')
        if col_name.lower().find(LOOKUP_SEP) >= 0:
            field_notes.append("Field renamed because it contained more than one '_' in a row.")
    if new_name.startswith('_'):
        new_name = 'field%s' % new_name
        field_notes.append("Field renamed because it started with '_'.")
    if new_name.endswith('_'):
        new_name = '%sfield' % new_name
        field_notes.append("Field renamed because it ended with '_'.")
    if keyword.iskeyword(new_name):
        new_name += '_field'
        field_notes.append('Field renamed because it was a Python reserved word.')
    if new_name[0].isdigit():
        new_name = 'number_%s' % new_name
        field_notes.append("Field renamed because it wasn't a valid Python identifier.")
    if new_name in used_column_names:
        num = 0
        while '%s_%d' % (new_name, num) in used_column_names:
            num += 1
        new_name = '%s_%d' % (new_name, num)
        field_notes.append('Field renamed because of name conflict.')
    if col_name != new_name and field_notes:
        field_params['db_column'] = col_name
    return (new_name, field_params, field_notes)