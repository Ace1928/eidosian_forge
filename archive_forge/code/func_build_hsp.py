from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
def build_hsp():
    if not query_tags and (not match_tags):
        raise ValueError(f'No data for query {query_id!r}, match {match_id!r}')
    assert query_tags, query_tags
    assert match_tags, match_tags
    evalue = align_tags.get('fa_expect')
    tool = global_tags.get('tool', '').upper()
    q = _extract_alignment_region(query_seq, query_tags)
    if tool in ['TFASTX'] and len(match_seq) == len(q):
        m = match_seq
    else:
        m = _extract_alignment_region(match_seq, match_tags)
    if len(q) != len(m):
        raise ValueError(f'Darn... amino acids vs nucleotide coordinates?\ntool: {tool}\nquery_seq: {query_seq}\nquery_tags: {query_tags}\n{q} length: {len(q)}\nmatch_seq: {match_seq}\nmatch_tags: {match_tags}\n{m} length: {len(m)}\nhandle.name: {handle.name}\n')
    annotations = {}
    records = []
    annotations.update(header_tags)
    annotations.update(align_tags)
    record = SeqRecord(Seq(q), id=query_id, name='query', description=query_descr, annotations={'original_length': int(query_tags['sq_len'])})
    record._al_start = int(query_tags['al_start'])
    record._al_stop = int(query_tags['al_stop'])
    if 'sq_type' in query_tags:
        if query_tags['sq_type'] == 'D':
            record.annotations['molecule_type'] = 'DNA'
        elif query_tags['sq_type'] == 'p':
            record.annotations['molecule_type'] = 'protein'
    records.append(record)
    record = SeqRecord(Seq(m), id=match_id, name='match', description=match_descr, annotations={'original_length': int(match_tags['sq_len'])})
    record._al_start = int(match_tags['al_start'])
    record._al_stop = int(match_tags['al_stop'])
    if 'sq_type' in match_tags:
        if match_tags['sq_type'] == 'D':
            record.annotations['molecule_type'] = 'DNA'
        elif match_tags['sq_type'] == 'p':
            record.annotations['molecule_type'] = 'protein'
    records.append(record)
    return MultipleSeqAlignment(records, annotations=annotations)