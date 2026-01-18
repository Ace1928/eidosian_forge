from abc import abstractmethod
from typing import (
from .config import registry
from .types import Floats2d, Ints1d
from .util import get_array_module, to_categorical
def convert_truths(self, truths, guesses: Floats2d) -> Tuple[Floats2d, Floats2d]:
    xp = get_array_module(guesses)
    missing = []
    negatives_mask = None
    if self.names:
        negatives_mask = xp.ones((len(truths), len(self.names)), dtype='f')
    missing_value = self.missing_value
    if isinstance(truths, list):
        truths = list(truths)
        if len(truths):
            if isinstance(truths[0], int):
                for i, value in enumerate(truths):
                    if value == missing_value:
                        missing.append(i)
            else:
                if self.names is None:
                    msg = "Cannot calculate loss from list of strings without names. You can pass the names as a keyword argument when you create the loss object, e.g. CategoricalCrossentropy(names=['dog', 'cat'])"
                    raise ValueError(msg)
                for i, value in enumerate(truths):
                    if value == missing_value:
                        truths[i] = self.names[0]
                        missing.append(i)
                    elif value and self.neg_prefix and value.startswith(self.neg_prefix):
                        truths[i] = value[len(self.neg_prefix):]
                        neg_index = self._name_to_i[truths[i]]
                        negatives_mask[i] = 0
                        negatives_mask[i][neg_index] = -1
                truths = [self._name_to_i[name] for name in truths]
        truths = xp.asarray(truths, dtype='i')
        mask = _make_mask(guesses, missing)
    else:
        mask = _make_mask_by_value(truths, guesses, missing_value)
    if truths.ndim != guesses.ndim:
        truths = to_categorical(cast(Ints1d, truths), n_classes=guesses.shape[-1], label_smoothing=self.label_smoothing)
    elif self.label_smoothing:
        raise ValueError('Label smoothing is only applied, when truths have type List[str], List[int] or Ints1d, but it seems like Floats2d was provided.')
    if negatives_mask is not None:
        truths *= negatives_mask
        truths[truths == -1] = 0
        negatives_mask[negatives_mask == -1] = 1
        mask *= negatives_mask
    return (truths, mask)