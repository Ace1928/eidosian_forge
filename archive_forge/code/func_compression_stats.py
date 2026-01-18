from collections import defaultdict
import logging
import math
from typing import Dict
import torch
import torch.distributed as dist
from . import default_hooks as default
from torch.distributed import distributed_c10d
def compression_stats(self):
    """
        Returns the latest compression statistics as a tuple of the form (compress_rate, numel_before_compression, numel_after_compression), where:

        compress_rate is the effective compression rate i.e. (number of elements before compression) / (number of elements after compression);

        numel_before_compression is the total number of elements before compression was applied; and,

        numel_after_compression is the total number of elements after compression was applied.
        """
    compress_rate = self.total_numel_before_compression / self.total_numel_after_compression if self.total_numel_after_compression > 0 else 0
    return (compress_rate, self.total_numel_before_compression, self.total_numel_after_compression)