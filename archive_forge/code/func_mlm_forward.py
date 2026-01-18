from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Optional, Tuple, cast
import numpy
from thinc.api import (
from thinc.loss import Loss
from thinc.types import Floats2d, Ints1d
from ...attrs import ID, ORTH
from ...errors import Errors
from ...util import OOV_RANK, registry
from ...vectors import Mode as VectorsMode
def mlm_forward(model, docs, is_train):
    mask, docs = _apply_mask(docs, random_words, mask_prob=mask_prob)
    mask = model.ops.asarray(mask).reshape((mask.shape[0], 1))
    output, backprop = model.layers[0](docs, is_train)

    def mlm_backward(d_output):
        d_output *= 1 - mask
        return backprop(d_output)
    return (output, mlm_backward)