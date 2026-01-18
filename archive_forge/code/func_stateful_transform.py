from functools import wraps
import numpy as np
from patsy.util import (atleast_2d_column_default,
def stateful_transform(class_):
    """Create a stateful transform callable object from a class that fulfills
    the :ref:`stateful transform protocol <stateful-transform-protocol>`.
    """

    @wraps(class_)
    def stateful_transform_wrapper(*args, **kwargs):
        transform = class_()
        transform.memorize_chunk(*args, **kwargs)
        transform.memorize_finish()
        return transform.transform(*args, **kwargs)
    stateful_transform_wrapper.__patsy_stateful_transform__ = class_
    return stateful_transform_wrapper