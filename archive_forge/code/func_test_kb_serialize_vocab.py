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
def test_kb_serialize_vocab(nlp):
    """Test serialization of the KB and custom strings"""
    entity = 'MyFunnyID'
    assert entity not in nlp.vocab.strings
    mykb = InMemoryLookupKB(nlp.vocab, entity_vector_length=1)
    assert not mykb.contains_entity(entity)
    mykb.add_entity(entity, freq=342, entity_vector=[3])
    assert mykb.contains_entity(entity)
    assert entity in mykb.vocab.strings
    with make_tempdir() as d:
        mykb.to_disk(d / 'kb')
        mykb_new = InMemoryLookupKB(Vocab(), entity_vector_length=1)
        mykb_new.from_disk(d / 'kb')
        assert entity in mykb_new.vocab.strings