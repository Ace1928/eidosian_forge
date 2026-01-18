from Bio.Align import Alignment
from Bio.Align import interfaces
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
def create_alignment():
    n = len(target_sequence)
    assert len(query_sequence) == n
    if n == 0:
        return
    coordinates = Alignment.infer_coordinates([target_sequence, query_sequence])
    coordinates[0, :] += target_start
    coordinates[1, :] += query_start
    sequence = {query_start: query_sequence.replace('-', '')}
    query_seq = Seq(sequence, length=query_length)
    query = SeqRecord(query_seq, id=self.query_name)
    sequence = {target_start: target_sequence.replace('-', '')}
    target_seq = Seq(sequence, length=target_length)
    target_annotations = {'hmm_name': hmm_name, 'hmm_description': hmm_description}
    target = SeqRecord(target_seq, id=target_name, annotations=target_annotations)
    fmt = f'{' ' * target_start}%-{target_length - target_start}s'
    target.letter_annotations['Consensus'] = fmt % target_consensus.replace('-', '')
    target.letter_annotations['ss_pred'] = fmt % target_ss_pred.replace('-', '')
    target.letter_annotations['ss_dssp'] = fmt % target_ss_dssp.replace('-', '')
    alignment_confidence = fmt % ''.join((c for t, c in zip(target_sequence, confidence) if t != '-'))
    fmt = f'{' ' * query_start}%-{query_length - query_start}s'
    if query_consensus:
        query.letter_annotations['Consensus'] = fmt % query_consensus.replace('-', '')
    if query_ss_pred:
        query.letter_annotations['ss_pred'] = fmt % query_ss_pred.replace('-', '')
    records = [target, query]
    alignment = Alignment(records, coordinates=coordinates)
    alignment.annotations = alignment_annotations
    alignment.column_annotations = {}
    alignment.column_annotations['column score'] = column_score
    alignment.column_annotations['Confidence'] = alignment_confidence
    return alignment