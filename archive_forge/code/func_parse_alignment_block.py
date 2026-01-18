import re
from itertools import chain
from ._base import (
from .exonerate_vulgar import _RE_VULGAR
def parse_alignment_block(self, header):
    """Parse alignment block, return query result, hits, hsps."""
    qresult = header['qresult']
    hit = header['hit']
    hsp = header['hsp']
    for val_name in ('query_start', 'query_end', 'hit_start', 'hit_end', 'query_strand', 'hit_strand'):
        assert val_name in hsp, hsp
    raw_aln_blocks, vulgar_comp = self._read_alignment()
    cmbn_rows = _stitch_rows(raw_aln_blocks)
    row_dict = _get_row_dict(len(cmbn_rows), qresult['model'])
    has_ner = 'NER' in qresult['model'].upper()
    seq_coords = _get_block_coords(cmbn_rows, row_dict, has_ner)
    tmp_seq_blocks = _get_blocks(cmbn_rows, seq_coords, row_dict)
    scodon_moves = _get_scodon_moves(tmp_seq_blocks)
    seq_blocks = _clean_blocks(tmp_seq_blocks)
    hsp['query_strand'] = _STRAND_MAP[hsp['query_strand']]
    hsp['hit_strand'] = _STRAND_MAP[hsp['hit_strand']]
    hsp['query_start'] = int(hsp['query_start'])
    hsp['query_end'] = int(hsp['query_end'])
    hsp['hit_start'] = int(hsp['hit_start'])
    hsp['hit_end'] = int(hsp['hit_end'])
    hsp['score'] = int(hsp['score'])
    hsp['query'] = [x['query'] for x in seq_blocks]
    hsp['hit'] = [x['hit'] for x in seq_blocks]
    hsp['aln_annotation'] = {}
    if 'protein2' in qresult['model'] or 'coding2' in qresult['model'] or '2protein' in qresult['model']:
        hsp['molecule_type'] = 'protein'
    for annot_type in ('similarity', 'query_annotation', 'hit_annotation'):
        try:
            hsp['aln_annotation'][annot_type] = [x[annot_type] for x in seq_blocks]
        except KeyError:
            pass
    if not has_ner:
        inter_coords = _get_inter_coords(seq_coords)
        inter_blocks = _get_blocks(cmbn_rows, inter_coords, row_dict)
        raw_inter_lens = re.findall(_RE_EXON_LEN, cmbn_rows[row_dict['midline']])
    for seq_type in ('query', 'hit'):
        if not has_ner:
            opp_type = 'hit' if seq_type == 'query' else 'query'
            inter_lens = _comp_intron_lens(seq_type, inter_blocks, raw_inter_lens)
        else:
            opp_type = seq_type
            inter_lens = [int(x) for x in re.findall(_RE_NER_LEN, cmbn_rows[row_dict[seq_type]])]
        if len(inter_lens) != len(hsp[opp_type]) - 1:
            raise ValueError('Length mismatch: %r vs %r' % (len(inter_lens), len(hsp[opp_type]) - 1))
        hsp['%s_ranges' % opp_type] = _comp_coords(hsp, opp_type, inter_lens)
        if not has_ner:
            hsp['%s_split_codons' % opp_type] = _comp_split_codons(hsp, opp_type, scodon_moves)
    for seq_type in ('query', 'hit'):
        if hsp['%s_strand' % seq_type] == -1:
            n_start = '%s_start' % seq_type
            n_end = '%s_end' % seq_type
            hsp[n_start], hsp[n_end] = (hsp[n_end], hsp[n_start])
    return {'qresult': qresult, 'hit': hit, 'hsp': hsp}