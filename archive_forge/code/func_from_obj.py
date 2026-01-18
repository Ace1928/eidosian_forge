import enum
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
@staticmethod
def from_obj(obj):
    """Tries to convert `obj` to a VariablePolicy instance."""
    if obj is None:
        return VariablePolicy.NONE
    if isinstance(obj, VariablePolicy):
        return obj
    key = str(obj).lower()
    for policy in VariablePolicy:
        if key == policy.value:
            return policy
    raise ValueError(f'Received invalid VariablePolicy value: {obj}.')