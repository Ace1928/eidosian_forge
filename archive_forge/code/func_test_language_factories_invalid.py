import pytest
from thinc.api import ConfigValidationError, Linear, Model
import spacy
from spacy.lang.de import German
from spacy.lang.en import English
from spacy.language import Language
from spacy.pipeline.tok2vec import DEFAULT_TOK2VEC_MODEL
from spacy.tokens import Doc
from spacy.util import SimpleFrozenDict, combine_score_weights, registry
from ..util import make_tempdir
def test_language_factories_invalid():
    """Test that assigning directly to Language.factories is now invalid and
    raises a custom error."""
    assert isinstance(Language.factories, SimpleFrozenDict)
    with pytest.raises(NotImplementedError):
        Language.factories['foo'] = 'bar'
    nlp = Language()
    assert isinstance(nlp.factories, SimpleFrozenDict)
    assert len(nlp.factories)
    with pytest.raises(NotImplementedError):
        nlp.factories['foo'] = 'bar'