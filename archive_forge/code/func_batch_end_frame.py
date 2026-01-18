import collections
from .utils import ExplicitEnum, is_torch_available, logging
def batch_end_frame(self):
    self.expand_frame(f'{self.prefix} *** Finished batch number={self.batch_number - 1} ***\n\n')