import warnings
from math import log
from Bio import BiopythonParserWarning
from Bio import BiopythonWarning
from Bio import StreamModeError
from Bio.File import as_handle
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import _clean
from .Interfaces import _get_seq_string
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
from .Interfaces import _TextIOSource
from typing import (
def as_fastq_solexa(record: SeqRecord) -> str:
    """Turn a SeqRecord into a Solexa FASTQ formatted string.

    This is used internally by the SeqRecord's .format("fastq-solexa")
    method and by the SeqIO.write(..., ..., "fastq-solexa") function.
    """
    seq_str = _get_seq_string(record)
    qualities_str = _get_solexa_quality_str(record)
    if len(qualities_str) != len(seq_str):
        raise ValueError('Record %s has sequence length %i but %i quality scores' % (record.id, len(seq_str), len(qualities_str)))
    id_ = _clean(record.id) if record.id else ''
    description = _clean(record.description)
    if description and description.split(None, 1)[0] == id_:
        title = description
    elif description:
        title = f'{id_} {description}'
    else:
        title = id_
    return f'@{title}\n{seq_str}\n+\n{qualities_str}\n'