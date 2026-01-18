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
def test_kb_serialization():
    vector_length = 3
    with make_tempdir() as tmp_dir:
        kb_dir = tmp_dir / 'kb'
        nlp1 = English()
        assert 'Q2146908' not in nlp1.vocab.strings
        mykb = InMemoryLookupKB(nlp1.vocab, entity_vector_length=vector_length)
        mykb.add_entity(entity='Q2146908', freq=12, entity_vector=[6, -4, 3])
        mykb.add_alias(alias='Russ Cochran', entities=['Q2146908'], probabilities=[0.8])
        assert 'Q2146908' in nlp1.vocab.strings
        mykb.to_disk(kb_dir)
        nlp2 = English()
        assert 'RandomWord' not in nlp2.vocab.strings
        nlp2.vocab.strings.add('RandomWord')
        assert 'RandomWord' in nlp2.vocab.strings
        assert 'Q2146908' not in nlp2.vocab.strings
        entity_linker = nlp2.add_pipe('entity_linker', last=True)
        entity_linker.set_kb(load_kb(kb_dir))
        assert 'Q2146908' in nlp2.vocab.strings
        assert 'RandomWord' in nlp2.vocab.strings