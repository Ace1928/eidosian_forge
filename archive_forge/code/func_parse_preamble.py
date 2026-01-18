import re
from Bio.SearchIO._utils import read_forward
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
from ._base import _BaseHmmerTextIndexer
def parse_preamble(self):
    """Parse HMMER2 preamble."""
    meta = {}
    state = 'GENERIC'
    while self.read_next():
        if state == 'GENERIC':
            if self.line.startswith('hmm'):
                meta['program'] = self.line.split('-')[0].strip()
            elif self.line.startswith('HMMER is'):
                continue
            elif self.line.startswith('HMMER'):
                meta['version'] = self.line.split()[1]
            elif self.line.count('-') == 36:
                state = 'OPTIONS'
            continue
        assert state == 'OPTIONS'
        assert 'program' in meta
        if self.line.count('-') == 32:
            break
        key, value = self.parse_key_value()
        if meta['program'] == 'hmmsearch':
            if key == 'Sequence database':
                meta['target'] = value
                continue
        elif meta['program'] == 'hmmpfam':
            if key == 'HMM file':
                meta['target'] = value
                continue
        meta[key] = value
    return meta