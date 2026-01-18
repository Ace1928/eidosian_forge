from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.six import string_types
def object_to_dict(obj, exclude=None):
    """
    Converts an object into a dict making the properties into keys, allows excluding certain keys
    """
    if exclude is None or not isinstance(exclude, list):
        exclude = []
    return dict(((key, getattr(obj, key)) for key in dir(obj) if not (key.startswith('_') or key in exclude)))