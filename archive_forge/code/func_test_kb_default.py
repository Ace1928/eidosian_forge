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
def test_kb_default(nlp):
    """Test that the default (empty) KB is loaded upon construction"""
    entity_linker = nlp.add_pipe('entity_linker', config={})
    assert len(entity_linker.kb) == 0
    with pytest.raises(ValueError, match='E139'):
        entity_linker.validate_kb()
    assert entity_linker.kb.get_size_entities() == 0
    assert entity_linker.kb.get_size_aliases() == 0
    assert entity_linker.kb.entity_vector_length == 64