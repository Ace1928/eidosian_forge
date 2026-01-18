import collections
from .utils import ExplicitEnum, is_torch_available, logging
def dump_saved_frames(self):
    print(f'\nDetected inf/nan during batch_number={self.batch_number}')
    print(f'Last {len(self.frames)} forward frames:')
    print(f'{'abs min':8} {'abs max':8} metadata')
    print('\n'.join(self.frames))
    print('\n\n')
    self.frames = []