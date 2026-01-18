from typing import Callable, List, Tuple
from thinc.api import Model, to_numpy
from thinc.types import Ints1d, Ragged
from ..util import registry
@registry.layers('spacy.extract_spans.v1')
def extract_spans() -> Model[Tuple[Ragged, Ragged], Ragged]:
    """Extract spans from a sequence of source arrays, as specified by an array
    of (start, end) indices. The output is a ragged array of the
    extracted spans.
    """
    return Model('extract_spans', forward, layers=[], refs={}, attrs={}, dims={}, init=init)