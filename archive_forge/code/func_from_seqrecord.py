import re
import warnings
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonWarning
from Bio.Phylo import BaseTree
@classmethod
def from_seqrecord(cls, record, is_aligned=None):
    """Create a new PhyloXML Sequence from a SeqRecord object."""
    if is_aligned is None:
        is_aligned = '-' in record.seq
    params = {'accession': Accession(record.id, ''), 'symbol': record.name, 'name': record.description, 'mol_seq': MolSeq(str(record.seq), is_aligned)}
    molecule_type = record.annotations.get('molecule_type')
    if molecule_type is not None:
        if 'DNA' in molecule_type:
            params['type'] = 'dna'
        elif 'RNA' in molecule_type:
            params['type'] = 'rna'
        elif 'protein' in molecule_type:
            params['type'] = 'protein'
    for key in ('id_ref', 'id_source', 'location'):
        if key in record.annotations:
            params[key] = record.annotations[key]
    if isinstance(record.annotations.get('uri'), dict):
        params['uri'] = Uri(**record.annotations['uri'])
    if record.annotations.get('annotations'):
        params['annotations'] = []
        for annot in record.annotations['annotations']:
            ann_args = {}
            for key in ('ref', 'source', 'evidence', 'type', 'desc'):
                if key in annot:
                    ann_args[key] = annot[key]
            if isinstance(annot.get('confidence'), list):
                ann_args['confidence'] = Confidence(*annot['confidence'])
            if isinstance(annot.get('properties'), list):
                ann_args['properties'] = [Property(**prop) for prop in annot['properties'] if isinstance(prop, dict)]
            params['annotations'].append(Annotation(**ann_args))
    if record.features:
        params['domain_architecture'] = DomainArchitecture(length=len(record.seq), domains=[ProteinDomain.from_seqfeature(feat) for feat in record.features])
    return Sequence(**params)