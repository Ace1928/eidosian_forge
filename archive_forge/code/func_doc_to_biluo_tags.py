import warnings
from typing import Dict, Iterable, Iterator, List, Tuple, Union, cast
from ..errors import Errors, Warnings
from ..tokens import Doc, Span
def doc_to_biluo_tags(doc: Doc, missing: str='O'):
    return offsets_to_biluo_tags(doc, [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents], missing=missing)