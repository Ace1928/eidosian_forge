import warnings
from typing import Dict, Iterable, Iterator, List, Tuple, Union, cast
from ..errors import Errors, Warnings
from ..tokens import Doc, Span
def biluo_tags_to_offsets(doc: Doc, tags: Iterable[str]) -> List[Tuple[int, int, Union[str, int]]]:
    """Encode per-token tags following the BILUO scheme into entity offsets.

    doc (Doc): The document that the BILUO tags refer to.
    tags (iterable): A sequence of BILUO tags with each tag describing one
        token. Each tags string will be of the form of either "", "O" or
        "{action}-{label}", where action is one of "B", "I", "L", "U".
    RETURNS (list): A sequence of `(start, end, label)` triples. `start` and
        `end` will be character-offset integers denoting the slice into the
        original string.
    """
    spans = biluo_tags_to_spans(doc, tags)
    return [(span.start_char, span.end_char, span.label_) for span in spans]