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
def get_vectors_loss(ops, docs, prediction, distance):
    """Compute a loss based on a distance between the documents' vectors and
    the prediction.
    """
    vocab = docs[0].vocab
    if vocab.vectors.mode == VectorsMode.default:
        ids = ops.flatten([doc.to_array(ID).ravel() for doc in docs])
        target = docs[0].vocab.vectors.data[ids]
        target[ids == OOV_RANK] = 0
        d_target, loss = distance(prediction, target)
    elif vocab.vectors.mode == VectorsMode.floret:
        keys = ops.flatten([cast(Ints1d, doc.to_array(ORTH)) for doc in docs])
        target = vocab.vectors.get_batch(keys)
        target = ops.as_contig(target)
        d_target, loss = distance(prediction, target)
    else:
        raise ValueError(Errors.E850.format(mode=vocab.vectors.mode))
    return (loss, d_target)