import re
import torch._C as C
def rawDispatchTable(self):
    return C._dispatch_dump_table(f'{self.namespace}::{self.name}')