import datetime
import re
from collections import namedtuple
from django.db.models.fields.related import RECURSIVE_RELATIONSHIP_CONSTANT
def field_references(model_tuple, field, reference_model_tuple, reference_field_name=None, reference_field=None):
    """
    Return either False or a FieldReference if `field` references provided
    context.

    False positives can be returned if `reference_field_name` is provided
    without `reference_field` because of the introspection limitation it
    incurs. This should not be an issue when this function is used to determine
    whether or not an optimization can take place.
    """
    remote_field = field.remote_field
    if not remote_field:
        return False
    references_to = None
    references_through = None
    if resolve_relation(remote_field.model, *model_tuple) == reference_model_tuple:
        to_fields = getattr(field, 'to_fields', None)
        if reference_field_name is None or to_fields is None or (None in to_fields and (reference_field is None or reference_field.primary_key)) or (reference_field_name in to_fields):
            references_to = (remote_field, to_fields)
    through = getattr(remote_field, 'through', None)
    if through and resolve_relation(through, *model_tuple) == reference_model_tuple:
        through_fields = remote_field.through_fields
        if reference_field_name is None or through_fields is None or reference_field_name in through_fields:
            references_through = (remote_field, through_fields)
    if not (references_to or references_through):
        return False
    return FieldReference(references_to, references_through)