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
@pytest.mark.issue(6730)
def test_issue6730(en_vocab):
    """Ensure that the KB does not accept empty strings, but otherwise IO works fine."""
    from spacy.kb.kb_in_memory import InMemoryLookupKB
    kb = InMemoryLookupKB(en_vocab, entity_vector_length=3)
    kb.add_entity(entity='1', freq=148, entity_vector=[1, 2, 3])
    with pytest.raises(ValueError):
        kb.add_alias(alias='', entities=['1'], probabilities=[0.4])
    assert kb.contains_alias('') is False
    kb.add_alias(alias='x', entities=['1'], probabilities=[0.2])
    kb.add_alias(alias='y', entities=['1'], probabilities=[0.1])
    with make_tempdir() as tmp_dir:
        kb.to_disk(tmp_dir)
        kb.from_disk(tmp_dir)
    assert kb.get_size_aliases() == 2
    assert set(kb.get_alias_strings()) == {'x', 'y'}