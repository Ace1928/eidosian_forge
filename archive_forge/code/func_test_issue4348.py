import pytest
from numpy.testing import assert_almost_equal, assert_equal
from thinc.api import compounding, get_current_ops
from spacy import util
from spacy.attrs import TAG
from spacy.lang.en import English
from spacy.language import Language
from spacy.training import Example
from ..util import make_tempdir
@pytest.mark.issue(4348)
def test_issue4348():
    """Test that training the tagger with empty data, doesn't throw errors"""
    nlp = English()
    example = Example.from_dict(nlp.make_doc(''), {'tags': []})
    TRAIN_DATA = [example, example]
    tagger = nlp.add_pipe('tagger')
    tagger.add_label('A')
    optimizer = nlp.initialize()
    for i in range(5):
        losses = {}
        batches = util.minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            nlp.update(batch, sgd=optimizer, losses=losses)