from typing import List
from aiokeydb.v1.exceptions import DataError
def keydb_args(self):
    args = [self.name]
    if self.as_name:
        args += [self.AS, self.as_name]
    args += self.args
    args += self.args_suffix
    return args