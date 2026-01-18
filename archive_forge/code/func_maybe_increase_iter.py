import logging
import torch
import torch.distributed as dist
from . import default_hooks as default
def maybe_increase_iter(self, bucket):
    if bucket.is_last():
        self.iter += 1
    if self.iter == self.start_localSGD_iter:
        logger.info('Start to apply local SGD after %s iterations.', self.iter)