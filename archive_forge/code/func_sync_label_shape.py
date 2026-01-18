import json
import logging
import random
import warnings
import numpy as np
from ..base import numeric_types
from .. import ndarray as nd
from ..ndarray._internal import _cvcopyMakeBorder as copyMakeBorder
from .. import io
from .image import RandomOrderAug, ColorJitterAug, LightingAug, ColorNormalizeAug
from .image import ResizeAug, ForceResizeAug, CastAug, HueJitterAug, RandomGrayAug
from .image import fixed_crop, ImageIter, Augmenter
from ..util import is_np_array
from .. import numpy as _mx_np  # pylint: disable=reimported
def sync_label_shape(self, it, verbose=False):
    """Synchronize label shape with the input iterator. This is useful when
        train/validation iterators have different label padding.

        Parameters
        ----------
        it : ImageDetIter
            The other iterator to synchronize
        verbose : bool
            Print verbose log if true

        Returns
        -------
        ImageDetIter
            The synchronized other iterator, the internal label shape is updated as well.

        Examples
        --------
        >>> train_iter = mx.image.ImageDetIter(32, (3, 300, 300), path_imgrec='train.rec')
        >>> val_iter = mx.image.ImageDetIter(32, (3, 300, 300), path.imgrec='val.rec')
        >>> train_iter.label_shape
        (30, 6)
        >>> val_iter.label_shape
        (25, 6)
        >>> val_iter = train_iter.sync_label_shape(val_iter, verbose=False)
        >>> train_iter.label_shape
        (30, 6)
        >>> val_iter.label_shape
        (30, 6)
        """
    assert isinstance(it, ImageDetIter), 'Synchronize with invalid iterator.'
    train_label_shape = self.label_shape
    val_label_shape = it.label_shape
    assert train_label_shape[1] == val_label_shape[1], 'object width mismatch.'
    max_count = max(train_label_shape[0], val_label_shape[0])
    if max_count > train_label_shape[0]:
        self.reshape(None, (max_count, train_label_shape[1]))
    if max_count > val_label_shape[0]:
        it.reshape(None, (max_count, val_label_shape[1]))
    if verbose and max_count > min(train_label_shape[0], val_label_shape[0]):
        logging.info('Resized label_shape to (%d, %d).', max_count, train_label_shape[1])
    return it