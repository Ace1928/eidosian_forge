import logging
import numpy as np
from .base import mx_real_t
from . import ndarray as nd
from .context import cpu
from .io import DataDesc
def load_data_batch(self, data_batch):
    """Load data and labels into arrays."""
    if self.sym_gen is not None:
        key = data_batch.bucket_key
        if key not in self.execgrp_bucket:
            symbol = self.sym_gen(key)
            execgrp = DataParallelExecutorGroup(symbol, self.arg_names, self.param_names, self.ctx, self.slices, data_batch, shared_group=self.execgrp)
            self.execgrp_bucket[key] = execgrp
        self.curr_execgrp = self.execgrp_bucket[key]
    else:
        self.curr_execgrp = self.execgrp
    self.curr_execgrp.load_data_batch(data_batch)