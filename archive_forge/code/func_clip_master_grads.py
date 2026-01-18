import math
from itertools import chain
from typing import Optional
import parlai.utils.logging as logging
from parlai.utils.misc import error_once
def clip_master_grads(self, gradient_clip):
    """
        Clips gradient norm and updates dynamic loss scaler.

        Returns -1 if the most recently computed gradients overflowed.
        """
    self._unscale_grads()
    grad_norm = clip_grad_norm(self.params, gradient_clip)
    overflow = has_overflow(grad_norm)
    self.scaler.update_scale(overflow)
    if overflow:
        if self.scaler.loss_scale <= self.min_loss_scale:
            raise FloatingPointError('Minimum loss scale reached ({}). Your loss is probably exploding. Try lowering the learning rate, using gradient clipping or increasing the batch size.'.format(self.min_loss_scale))
        logging.info(f'Overflow: setting loss scale to {self.scaler.loss_scale}')
        self.zero_grad()
        return -1
    return grad_norm