from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import _clean
from .Interfaces import _get_seq_string
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
def as_tab(record):
    """Return record as tab separated (id(tab)seq) string."""
    title = _clean(record.id)
    seq = _get_seq_string(record)
    assert '\t' not in title
    assert '\n' not in title
    assert '\r' not in title
    assert '\t' not in seq
    assert '\n' not in seq
    assert '\r' not in seq
    return f'{title}\t{seq}\n'