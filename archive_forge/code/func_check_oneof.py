import collections
import collections.abc
import copy
import inspect
from cloudsdk.google.protobuf import field_mask_pb2
from cloudsdk.google.protobuf import message
from cloudsdk.google.protobuf import wrappers_pb2
def check_oneof(**kwargs):
    """Raise ValueError if more than one keyword argument is not ``None``.

    Args:
        kwargs (dict): The keyword arguments sent to the function.

    Raises:
        ValueError: If more than one entry in ``kwargs`` is not ``None``.
    """
    if not kwargs:
        return
    not_nones = [val for val in kwargs.values() if val is not None]
    if len(not_nones) > 1:
        raise ValueError('Only one of {fields} should be set.'.format(fields=', '.join(sorted(kwargs.keys()))))