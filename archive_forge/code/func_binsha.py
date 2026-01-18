from gitdb.util import bin_to_hex
from gitdb.fun import (
@property
def binsha(self):
    return self[0]