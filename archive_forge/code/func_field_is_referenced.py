import datetime
import re
from collections import namedtuple
from django.db.models.fields.related import RECURSIVE_RELATIONSHIP_CONSTANT
def field_is_referenced(state, model_tuple, field_tuple):
    """Return whether `field_tuple` is referenced by any state models."""
    return next(get_references(state, model_tuple, field_tuple), None) is not None