import warnings
import re
import sys
from collections import defaultdict
from Bio.File import as_handle
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonParserWarning
from typing import List
def parse_footer(self):
    """Return a tuple containing a list of any misc strings, and the sequence."""
    if self.line[:self.HEADER_WIDTH].rstrip() not in self.SEQUENCE_HEADERS:
        raise ValueError(f"Footer format unexpected:  '{self.line}'")
    misc_lines = []
    while self.line[:self.HEADER_WIDTH].rstrip() in self.SEQUENCE_HEADERS or self.line[:self.HEADER_WIDTH] == ' ' * self.HEADER_WIDTH or 'WGS' == self.line[:3]:
        misc_lines.append(self.line.rstrip())
        self.line = self.handle.readline()
        if not self.line:
            raise ValueError('Premature end of file')
    if self.line[:self.HEADER_WIDTH].rstrip() in self.SEQUENCE_HEADERS:
        raise ValueError(f"Eh? '{self.line}'")
    seq_lines = []
    line = self.line
    while True:
        if not line:
            warnings.warn('Premature end of file in sequence data', BiopythonParserWarning)
            line = '//'
            break
        line = line.rstrip()
        if not line:
            warnings.warn('Blank line in sequence data', BiopythonParserWarning)
            line = self.handle.readline()
            continue
        if line == '//':
            break
        if line.startswith('CONTIG'):
            break
        if len(line) > 9 and line[9:10] != ' ':
            warnings.warn('Invalid indentation for sequence line', BiopythonParserWarning)
            line = line[1:]
            if len(line) > 9 and line[9:10] != ' ':
                raise ValueError(f"Sequence line mal-formed, '{line}'")
        seq_lines.append(line[10:])
        line = self.handle.readline()
    self.line = line
    return (misc_lines, ''.join(seq_lines).replace(' ', ''))