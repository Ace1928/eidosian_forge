from typing import List, Union
def build_args(self):
    args = [str(self.cid)]
    if self.max_idle:
        args += ['MAXIDLE', str(self.max_idle)]
    if self.count:
        args += ['COUNT', str(self.count)]
    return args