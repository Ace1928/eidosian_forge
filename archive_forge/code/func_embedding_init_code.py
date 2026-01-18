import sys, types
from .lock import allocate_lock
from .error import CDefError
from . import model
def embedding_init_code(self, pysource):
    if self._embedding:
        raise ValueError('embedding_init_code() can only be called once')
    import re
    match = re.match('\\s*\\n', pysource)
    if match:
        pysource = pysource[match.end():]
    lines = pysource.splitlines() or ['']
    prefix = re.match('\\s*', lines[0]).group()
    for i in range(1, len(lines)):
        line = lines[i]
        if line.rstrip():
            while not line.startswith(prefix):
                prefix = prefix[:-1]
    i = len(prefix)
    lines = [line[i:] + '\n' for line in lines]
    pysource = ''.join(lines)
    compile(pysource, 'cffi_init', 'exec')
    self._embedding = pysource