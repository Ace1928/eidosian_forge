import os
from ....branch import Branch
from ....tests.blackbox import ExternalBase
def make_file(self, name, contents):
    with open(name, 'w' + ('b' if isinstance(contents, bytes) else '')) as f:
        f.write(contents)