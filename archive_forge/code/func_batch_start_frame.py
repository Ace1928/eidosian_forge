import collections
from .utils import ExplicitEnum, is_torch_available, logging
def batch_start_frame(self):
    self.expand_frame(f'\n\n{self.prefix} *** Starting batch number={self.batch_number} ***')
    self.expand_frame(f'{'abs min':8} {'abs max':8} metadata')