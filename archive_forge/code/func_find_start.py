import warnings
import re
import sys
from collections import defaultdict
from Bio.File import as_handle
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonParserWarning
from typing import List
def find_start(self):
    """Read in lines until find the ID/LOCUS line, which is returned.

        Any preamble (such as the header used by the NCBI on ``*.seq.gz`` archives)
        will we ignored.
        """
    while True:
        if self.line:
            line = self.line
            self.line = ''
        else:
            line = self.handle.readline()
        if not line:
            if self.debug:
                print('End of file')
            return None
        if isinstance(line[0], int):
            raise ValueError('Is this handle in binary mode not text mode?')
        if line[:self.HEADER_WIDTH] == self.RECORD_START:
            if self.debug > 1:
                print('Found the start of a record:\n' + line)
            break
        line = line.rstrip()
        if line == '//':
            if self.debug > 1:
                print('Skipping // marking end of last record')
        elif line == '':
            if self.debug > 1:
                print('Skipping blank line before record')
        elif self.debug > 1:
            print('Skipping header line before record:\n' + line)
    self.line = line
    return line