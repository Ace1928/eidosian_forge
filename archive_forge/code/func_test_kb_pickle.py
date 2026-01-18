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
@pytest.mark.xfail(reason='Needs fixing')
def test_kb_pickle():
    nlp = English()
    kb_1 = InMemoryLookupKB(nlp.vocab, entity_vector_length=3)
    kb_1.add_entity(entity='Q2146908', freq=12, entity_vector=[6, -4, 3])
    assert not kb_1.contains_alias('Russ Cochran')
    kb_1.add_alias(alias='Russ Cochran', entities=['Q2146908'], probabilities=[0.8])
    assert kb_1.contains_alias('Russ Cochran')
    data = pickle.dumps(kb_1)
    kb_2 = pickle.loads(data)
    assert kb_2.contains_alias('Russ Cochran')