import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from ..utils.generic import ModelOutput
def loader_batch_item(self):
    """
        Return item located at `loader_batch_index` within the current `loader_batch_data`.
        """
    if isinstance(self._loader_batch_data, torch.Tensor):
        result = self._loader_batch_data[self._loader_batch_index]
    else:
        loader_batched = {}
        for k, element in self._loader_batch_data.items():
            if isinstance(element, ModelOutput):
                element = element.to_tuple()
                if isinstance(element[0], torch.Tensor):
                    loader_batched[k] = tuple((el[self._loader_batch_index].unsqueeze(0) for el in element))
                elif isinstance(element[0], np.ndarray):
                    loader_batched[k] = tuple((np.expand_dims(el[self._loader_batch_index], 0) for el in element))
                continue
            if k in {'hidden_states', 'past_key_values', 'attentions'} and isinstance(element, tuple):
                if isinstance(element[0], torch.Tensor):
                    loader_batched[k] = tuple((el[self._loader_batch_index].unsqueeze(0) for el in element))
                elif isinstance(element[0], np.ndarray):
                    loader_batched[k] = tuple((np.expand_dims(el[self._loader_batch_index], 0) for el in element))
                continue
            if element is None:
                loader_batched[k] = None
            elif isinstance(element[self._loader_batch_index], torch.Tensor):
                loader_batched[k] = element[self._loader_batch_index].unsqueeze(0)
            elif isinstance(element[self._loader_batch_index], np.ndarray):
                loader_batched[k] = np.expand_dims(element[self._loader_batch_index], 0)
            else:
                loader_batched[k] = element[self._loader_batch_index]
        result = self._loader_batch_data.__class__(loader_batched)
    self._loader_batch_index += 1
    return result