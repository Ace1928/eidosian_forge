import re
from Bio.SearchIO._utils import read_forward
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
from ._base import _BaseHmmerTextIndexer
def parse_hsp_alignments(self):
    """Parse a HMMER2 HSP alignment block."""
    if not self.line.startswith('Alignments'):
        return
    while self.read_next():
        if self.line == '//' or self.line.startswith('Histogram'):
            break
        match = re.search(_HSP_ALIGN_LINE, self.line)
        if match is None:
            continue
        id_ = match.group(1)
        idx = int(match.group(2))
        num = int(match.group(3))
        hit = self.qresult[id_]
        if hit.domain_obs_num != num:
            continue
        frag = hit[idx - 1][0]
        hmmseq = ''
        consensus = ''
        otherseq = ''
        structureseq = ''
        pad = 0
        while self.read_next() and self.line.startswith(' '):
            if self.line[16:18] == 'CS':
                structureseq += self.line[19:].strip()
                if not self.read_next():
                    break
            if self.line[19:22] == '*->':
                seq = self.line[22:]
                pad = 3
            else:
                seq = self.line[19:]
                pad = 0
            hmmseq += seq
            line_len = len(seq)
            if not self.read_next(rstrip=False):
                break
            consensus += self.line[19 + pad:19 + pad + line_len]
            extra_padding = len(hmmseq) - len(consensus)
            consensus += ' ' * extra_padding
            if not self.read_next():
                break
            parts = self.line[19:].split()
            if len(parts) == 2:
                otherseq += self.line[19:].split()[0].strip()
        self.push_back(self.line)
        if hmmseq.endswith('<-*'):
            hmmseq = hmmseq[:-3]
            consensus = consensus[:-3]
        frag.aln_annotation['similarity'] = consensus
        if structureseq:
            frag.aln_annotation['CS'] = structureseq
        if self._meta['program'] == 'hmmpfam':
            frag.hit = hmmseq
            frag.query = otherseq
        else:
            frag.hit = otherseq
            frag.query = hmmseq