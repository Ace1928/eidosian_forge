import math
from itertools import chain
from typing import Optional
import parlai.utils.logging as logging
from parlai.utils.misc import error_once
def multiply_grads(self, c):
    """
        Multiplies grads by a constant `c`.
        """
    if self._grads_are_scaled:
        self._unscale_grads(c)
    else:
        for p in self.params:
            if p.grad is not None:
                p.grad.data.mul_(c)