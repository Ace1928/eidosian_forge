import numpy as np
from . import _translation_utils as translation_utils
from .... import symbol
def lesser(attrs, inputs, proto_obj):
    """Logical Lesser operator with broadcasting."""
    return ('broadcast_lesser', attrs, inputs)