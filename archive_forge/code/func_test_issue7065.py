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
@pytest.mark.issue(7065)
def test_issue7065():
    text = "Kathleen Battle sang in Mahler 's Symphony No. 8 at the Cincinnati Symphony Orchestra 's May Festival."
    nlp = English()
    nlp.add_pipe('sentencizer')
    ruler = nlp.add_pipe('entity_ruler')
    patterns = [{'label': 'THING', 'pattern': [{'LOWER': 'symphony'}, {'LOWER': 'no'}, {'LOWER': '.'}, {'LOWER': '8'}]}]
    ruler.add_patterns(patterns)
    doc = nlp(text)
    sentences = [s for s in doc.sents]
    assert len(sentences) == 2
    sent0 = sentences[0]
    ent = doc.ents[0]
    assert ent.start < sent0.end < ent.end
    assert sentences.index(ent.sent) == 0