from functools import lru_cache
import torch
from parlai.core.torch_ranker_agent import TorchRankerAgent
from .modules import MemNN, opt_to_kwargs
def build_dictionary(self):
    """
        Add the time features to the dictionary before building the model.
        """
    d = super().build_dictionary()
    if self.use_time_features:
        for i in range(self.memsize):
            d[self._time_feature(i)] = 100000000 + i
    return d