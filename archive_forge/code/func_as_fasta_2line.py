import warnings
from typing import Callable, Optional, Tuple
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import _clean
from .Interfaces import _get_seq_string
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
from .Interfaces import _TextIOSource
def as_fasta_2line(record):
    """Turn a SeqRecord into a two-line FASTA formatted string.

    This is used internally by the SeqRecord's .format("fasta-2line")
    method and by the SeqIO.write(..., ..., "fasta-2line") function.
    """
    id = _clean(record.id)
    description = _clean(record.description)
    if description and description.split(None, 1)[0] == id:
        title = description
    elif description:
        title = f'{id} {description}'
    else:
        title = id
    assert '\n' not in title
    assert '\r' not in title
    data = _get_seq_string(record)
    assert '\n' not in data
    assert '\r' not in data
    return f'>{title}\n{data}\n'