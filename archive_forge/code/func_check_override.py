from ray.rllib.utils.deprecation import Deprecated
from ray.util.annotations import _mark_annotated
def check_override(method):
    if method.__name__ not in dir(cls):
        raise NameError('{} does not override any method of {}'.format(method, cls))
    return method