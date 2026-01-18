from typing import Any, Callable, Dict, Iterable, Tuple
import pytest
from numpy.testing import assert_equal
from spacy import Language, registry, util
from spacy.attrs import ENT_KB_ID
from spacy.compat import pickle
from spacy.kb import Candidate, InMemoryLookupKB, KnowledgeBase, get_candidates
from spacy.lang.en import English
from spacy.ml import load_kb
from spacy.ml.models.entity_linker import build_span_maker
from spacy.pipeline import EntityLinker
from spacy.pipeline.legacy import EntityLinker_v1
from spacy.pipeline.tok2vec import DEFAULT_TOK2VEC_MODEL
from spacy.scorer import Scorer
from spacy.tests.util import make_tempdir
from spacy.tokens import Doc, Span
from spacy.training import Example
from spacy.util import ensure_path
from spacy.vocab import Vocab
def test_kb_to_bytes():
    nlp = English()
    kb_1 = InMemoryLookupKB(nlp.vocab, entity_vector_length=3)
    kb_1.add_entity(entity='Q2146908', freq=12, entity_vector=[6, -4, 3])
    kb_1.add_entity(entity='Q66', freq=9, entity_vector=[1, 2, 3])
    kb_1.add_alias(alias='Russ Cochran', entities=['Q2146908'], probabilities=[0.8])
    kb_1.add_alias(alias='Boeing', entities=['Q66'], probabilities=[0.5])
    kb_1.add_alias(alias='Randomness', entities=['Q66', 'Q2146908'], probabilities=[0.1, 0.2])
    assert kb_1.contains_alias('Russ Cochran')
    kb_bytes = kb_1.to_bytes()
    kb_2 = InMemoryLookupKB(nlp.vocab, entity_vector_length=3)
    assert not kb_2.contains_alias('Russ Cochran')
    kb_2 = kb_2.from_bytes(kb_bytes)
    assert kb_1.get_size_entities() == kb_2.get_size_entities()
    assert kb_1.entity_vector_length == kb_2.entity_vector_length
    assert kb_1.get_entity_strings() == kb_2.get_entity_strings()
    assert kb_1.get_vector('Q2146908') == kb_2.get_vector('Q2146908')
    assert kb_1.get_vector('Q66') == kb_2.get_vector('Q66')
    assert kb_2.contains_alias('Russ Cochran')
    assert kb_1.get_size_aliases() == kb_2.get_size_aliases()
    assert kb_1.get_alias_strings() == kb_2.get_alias_strings()
    assert len(kb_1.get_alias_candidates('Russ Cochran')) == len(kb_2.get_alias_candidates('Russ Cochran'))
    assert len(kb_1.get_alias_candidates('Randomness')) == len(kb_2.get_alias_candidates('Randomness'))