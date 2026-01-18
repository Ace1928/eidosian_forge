import warnings
import re
import sys
from collections import defaultdict
from Bio.File import as_handle
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonParserWarning
from typing import List
def parse_features(self, skip=False):
    """Return list of tuples for the features (if present).

        Each feature is returned as a tuple (key, location, qualifiers)
        where key and location are strings (e.g. "CDS" and
        "complement(join(490883..490885,1..879))") while qualifiers
        is a list of two string tuples (feature qualifier keys and values).

        Assumes you have already read to the start of the features table.
        """
    if self.line.rstrip() not in self.FEATURE_START_MARKERS:
        if self.debug:
            print("Didn't find any feature table")
        return []
    while self.line.rstrip() in self.FEATURE_START_MARKERS:
        self.line = self.handle.readline()
    bad_position_re = re.compile('([0-9]+)>')
    features = []
    line = self.line
    while True:
        if not line:
            raise ValueError('Premature end of line during features table')
        if line[:self.HEADER_WIDTH].rstrip() in self.SEQUENCE_HEADERS:
            if self.debug:
                print('Found start of sequence')
            break
        line = line.rstrip()
        if line == '//':
            raise ValueError("Premature end of features table, marker '//' found")
        if line in self.FEATURE_END_MARKERS:
            if self.debug:
                print('Found end of features')
            line = self.handle.readline()
            break
        if line[2:self.FEATURE_QUALIFIER_INDENT].strip() == '':
            line = self.handle.readline()
            continue
        if skip:
            line = self.handle.readline()
            while line[:self.FEATURE_QUALIFIER_INDENT] == self.FEATURE_QUALIFIER_SPACER:
                line = self.handle.readline()
        else:
            assert line[:2] == 'FT'
            try:
                feature_key, location_start = line[2:].strip().split()
            except ValueError:
                feature_key = line[2:25].strip()
                location_start = line[25:].strip()
            feature_lines = [location_start]
            line = self.handle.readline()
            while line[:self.FEATURE_QUALIFIER_INDENT] == self.FEATURE_QUALIFIER_SPACER or line.rstrip() == '':
                assert line[:2] == 'FT'
                feature_lines.append(line[self.FEATURE_QUALIFIER_INDENT:].strip())
                line = self.handle.readline()
            feature_key, location, qualifiers = self.parse_feature(feature_key, feature_lines)
            if '>' in location:
                location = bad_position_re.sub('>\\1', location)
            features.append((feature_key, location, qualifiers))
    self.line = line
    return features