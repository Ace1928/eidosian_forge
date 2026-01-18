import collections
import collections.abc
import copy
import inspect
from cloudsdk.google.protobuf import field_mask_pb2
from cloudsdk.google.protobuf import message
from cloudsdk.google.protobuf import wrappers_pb2
def get_messages(module):
    """Discovers all protobuf Message classes in a given import module.

    Args:
        module (module): A Python module; :func:`dir` will be run against this
            module to find Message subclasses.

    Returns:
        dict[str, google.protobuf.message.Message]: A dictionary with the
            Message class names as keys, and the Message subclasses themselves
            as values.
    """
    answer = collections.OrderedDict()
    for name in dir(module):
        candidate = getattr(module, name)
        if inspect.isclass(candidate) and issubclass(candidate, message.Message):
            answer[name] = candidate
    return answer