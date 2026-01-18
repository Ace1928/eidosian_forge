from __future__ import absolute_import
import os
from pickle import PicklingError
from ray.cloudpickle.cloudpickle import *  # noqa
from ray.cloudpickle.cloudpickle_fast import CloudPickler, dumps, dump  # noqa
def dumps_debug(obj, *args, **kwargs):
    try:
        return dumps(obj, *args, **kwargs)
    except (TypeError, PicklingError) as exc:
        if os.environ.get('RAY_PICKLE_VERBOSE_DEBUG'):
            from ray.util.check_serialize import inspect_serializability
            inspect_serializability(obj)
            raise
        else:
            msg = _warn_msg(obj, 'ray.cloudpickle.dumps', exc)
            raise type(exc)(msg)