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
def test_kb_valid_entities(nlp):
    """Test the valid construction of a KB with 3 entities and two aliases"""
    mykb = InMemoryLookupKB(nlp.vocab, entity_vector_length=3)
    mykb.add_entity(entity='Q1', freq=19, entity_vector=[8, 4, 3])
    mykb.add_entity(entity='Q2', freq=5, entity_vector=[2, 1, 0])
    mykb.add_entity(entity='Q3', freq=25, entity_vector=[-1, -6, 5])
    mykb.add_alias(alias='douglas', entities=['Q2', 'Q3'], probabilities=[0.8, 0.2])
    mykb.add_alias(alias='adam', entities=['Q2'], probabilities=[0.9])
    assert mykb.get_size_entities() == 3
    assert mykb.get_size_aliases() == 2
    assert mykb.get_vector('Q1') == [8, 4, 3]
    assert mykb.get_vector('Q2') == [2, 1, 0]
    assert mykb.get_vector('Q3') == [-1, -6, 5]
    assert_almost_equal(mykb.get_prior_prob(entity='Q2', alias='douglas'), 0.8)
    assert_almost_equal(mykb.get_prior_prob(entity='Q3', alias='douglas'), 0.2)
    assert_almost_equal(mykb.get_prior_prob(entity='Q342', alias='douglas'), 0.0)
    assert_almost_equal(mykb.get_prior_prob(entity='Q3', alias='douglassssss'), 0.0)