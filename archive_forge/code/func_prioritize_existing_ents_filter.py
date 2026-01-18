import warnings
from functools import partial
from pathlib import Path
from typing import (
import srsly
from .. import util
from ..errors import Errors, Warnings
from ..language import Language
from ..matcher import Matcher, PhraseMatcher
from ..matcher.levenshtein import levenshtein_compare
from ..scorer import Scorer
from ..tokens import Doc, Span
from ..training import Example
from ..util import SimpleFrozenList, ensure_path, registry
from .pipe import Pipe
def prioritize_existing_ents_filter(entities: Iterable[Span], spans: Iterable[Span]) -> List[Span]:
    """Merge entities and spans into one list without overlaps by prioritizing
    existing entities. Intended to replicate the overwrite_ents=False behavior
    from the EntityRuler.

    entities (Iterable[Span]): The entities, already filtered for overlaps.
    spans (Iterable[Span]): The spans to merge, may contain overlaps.
    RETURNS (List[Span]): Filtered list of non-overlapping spans.
    """
    get_sort_key = lambda span: (span.end - span.start, -span.start)
    spans = sorted(spans, key=get_sort_key, reverse=True)
    entities = list(entities)
    new_entities = []
    seen_tokens: Set[int] = set()
    seen_tokens.update(*(range(ent.start, ent.end) for ent in entities))
    for span in spans:
        start = span.start
        end = span.end
        if all((token.i not in seen_tokens for token in span)):
            new_entities.append(span)
            seen_tokens.update(range(start, end))
    return entities + new_entities