import collections
import functools
import re
import threading
import warnings
import numpy as np
import wrapt
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import pywrap_tf_session as tf_session
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import device
from tensorflow.python.framework import error_interpolation
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import stack
from tensorflow.python.framework import tensor
from tensorflow.python.ops import session_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.experimental import mixed_precision_global_state
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
@staticmethod
def for_fetch(fetch):
    """Creates fetch mapper that handles the structure of `fetch`.

    The default graph must be the one from which we want to fetch values when
    this function is called.

    Args:
      fetch: An arbitrary fetch structure: singleton, list, tuple, namedtuple,
        or dict.

    Returns:
      An instance of a subclass of `_FetchMapper` that handles the shape.
    """
    if fetch is None:
        raise TypeError(f'Argument `fetch` = {fetch} has invalid type "{type(fetch).__name__}". Cannot be None')
    elif isinstance(fetch, (list, tuple)):
        return _ListFetchMapper(fetch)
    elif isinstance(fetch, collections_abc.Mapping):
        return _DictFetchMapper(fetch)
    elif _is_attrs_instance(fetch):
        return _AttrsFetchMapper(fetch)
    else:
        for tensor_type, fetch_fn, _, _ in _REGISTERED_EXPANSIONS:
            if isinstance(fetch, tensor_type):
                fetches, contraction_fn = fetch_fn(fetch)
                return _ElementFetchMapper(fetches, contraction_fn)
    raise TypeError(f'Argument `fetch` = {fetch} has invalid type "{type(fetch).__name__}"')