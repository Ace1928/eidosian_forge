import re
from Bio.SearchIO._utils import read_forward
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
from ._base import _BaseHmmerTextIndexer
def parse_qresult(self):
    """Parse a HMMER2 query block."""
    while self.read_next():
        if not self.line.startswith('Query'):
            return
        _, id_ = self.parse_key_value()
        self.qresult = QueryResult(id=id_)
        description = None
        while self.read_next() and (not self.line.startswith('Scores')):
            if self.line.startswith('Accession'):
                self.qresult.accession = self.parse_key_value()[1]
            if self.line.startswith('Description'):
                description = self.parse_key_value()[1]
        hit_placeholders = self.parse_hits()
        if len(hit_placeholders) > 0:
            self.parse_hsps(hit_placeholders)
            self.parse_hsp_alignments()
        while not self.line.startswith('Query'):
            self.read_next()
            if not self.line:
                break
        self.buf.append(self.line)
        if description is not None:
            self.qresult.description = description
        yield self.qresult