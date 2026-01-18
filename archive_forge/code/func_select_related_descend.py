import functools
import inspect
import logging
from collections import namedtuple
from django.core.exceptions import FieldError
from django.db import DEFAULT_DB_ALIAS, DatabaseError, connections
from django.db.models.constants import LOOKUP_SEP
from django.utils import tree
from django.utils.functional import cached_property
from django.utils.hashable import make_hashable
def select_related_descend(field, restricted, requested, select_mask, reverse=False):
    """
    Return True if this field should be used to descend deeper for
    select_related() purposes. Used by both the query construction code
    (compiler.get_related_selections()) and the model instance creation code
    (compiler.klass_info).

    Arguments:
     * field - the field to be checked
     * restricted - a boolean field, indicating if the field list has been
       manually restricted using a requested clause)
     * requested - The select_related() dictionary.
     * select_mask - the dictionary of selected fields.
     * reverse - boolean, True if we are checking a reverse select related
    """
    if not field.remote_field:
        return False
    if field.remote_field.parent_link and (not reverse):
        return False
    if restricted:
        if reverse and field.related_query_name() not in requested:
            return False
        if not reverse and field.name not in requested:
            return False
    if not restricted and field.null:
        return False
    if restricted and select_mask and (field.name in requested) and (field not in select_mask):
        raise FieldError(f'Field {field.model._meta.object_name}.{field.name} cannot be both deferred and traversed using select_related at the same time.')
    return True