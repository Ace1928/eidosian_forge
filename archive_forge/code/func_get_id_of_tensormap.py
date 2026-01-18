from __future__ import annotations
from ..runtime import driver
def get_id_of_tensormap(self):
    return self.TMADescArgIdx + len(self.ids_of_folded_args)