import logging
from math import cos, pi
def get_warmup_lr(self, num_update):
    assert num_update < self.warmup_steps
    if self.warmup_mode == 'linear':
        increase = (self.warmup_final_lr - self.warmup_begin_lr) * float(num_update) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase
    elif self.warmup_mode == 'constant':
        return self.warmup_begin_lr
    else:
        raise ValueError('Invalid warmup mode %s' % self.warmup_mode)