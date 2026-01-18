import random
import numpy
import pytest
import srsly
from thinc.api import Adam, compounding
import spacy
from spacy.lang.en import English
from spacy.tokens import Doc, DocBin
from spacy.training import (
from spacy.training.align import get_alignments
from spacy.training.alignment_array import AlignmentArray
from spacy.training.converters import json_to_docs
from spacy.training.loop import train_while_improving
from spacy.util import (
from ..util import make_tempdir
@pytest.mark.issue(7029)
def test_issue7029():
    """Test that an empty document doesn't mess up an entire batch."""
    TRAIN_DATA = [('I like green eggs', {'tags': ['N', 'V', 'J', 'N']}), ('Eat blue ham', {'tags': ['V', 'J', 'N']})]
    nlp = English.from_config(load_config_from_str(CONFIG_7029))
    train_examples = []
    for t in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))
    optimizer = nlp.initialize(get_examples=lambda: train_examples)
    for i in range(50):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)
    texts = ['first', 'second', 'third', 'fourth', 'and', 'then', 'some', '']
    docs1 = list(nlp.pipe(texts, batch_size=1))
    docs2 = list(nlp.pipe(texts, batch_size=4))
    assert [doc[0].tag_ for doc in docs1[:-1]] == [doc[0].tag_ for doc in docs2[:-1]]