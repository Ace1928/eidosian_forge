import numpy
import pytest
from spacy import registry, util
from spacy.lang.en import English
from spacy.pipeline import AttributeRuler
from spacy.tokens import Doc
from spacy.training import Example
from ..util import make_tempdir
def test_attributeruler_init_clear(nlp, pattern_dicts):
    """Test that initialization clears patterns."""
    ruler = nlp.add_pipe('attribute_ruler')
    assert not len(ruler.matcher)
    ruler.add_patterns(pattern_dicts)
    assert len(ruler.matcher)
    ruler.initialize(lambda: [])
    assert not len(ruler.matcher)