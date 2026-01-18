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
@Language.factory('future_entity_ruler', assigns=['doc.ents'], default_config={'phrase_matcher_attr': None, 'validate': False, 'overwrite_ents': False, 'scorer': {'@scorers': 'spacy.entity_ruler_scorer.v1'}, 'ent_id_sep': '__unused__', 'matcher_fuzzy_compare': {'@misc': 'spacy.levenshtein_compare.v1'}}, default_score_weights={'ents_f': 1.0, 'ents_p': 0.0, 'ents_r': 0.0, 'ents_per_type': None})
def make_entity_ruler(nlp: Language, name: str, phrase_matcher_attr: Optional[Union[int, str]], matcher_fuzzy_compare: Callable, validate: bool, overwrite_ents: bool, scorer: Optional[Callable], ent_id_sep: str):
    if overwrite_ents:
        ents_filter = prioritize_new_ents_filter
    else:
        ents_filter = prioritize_existing_ents_filter
    return SpanRuler(nlp, name, spans_key=None, spans_filter=None, annotate_ents=True, ents_filter=ents_filter, phrase_matcher_attr=phrase_matcher_attr, matcher_fuzzy_compare=matcher_fuzzy_compare, validate=validate, overwrite=False, scorer=scorer)