import pytest
from numpy.testing import assert_almost_equal, assert_equal
from thinc.api import get_current_ops
from spacy import util
from spacy.attrs import MORPH
from spacy.lang.en import English
from spacy.language import Language
from spacy.morphology import Morphology
from spacy.tests.util import make_tempdir
from spacy.tokens import Doc
from spacy.training import Example
def test_label_smoothing():
    nlp = Language()
    morph_no_ls = nlp.add_pipe('morphologizer', 'no_label_smoothing')
    morph_ls = nlp.add_pipe('morphologizer', 'label_smoothing', config=dict(label_smoothing=0.05))
    train_examples = []
    losses = {}
    for tag in TAGS:
        morph_no_ls.add_label(tag)
        morph_ls.add_label(tag)
    for t in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))
    nlp.initialize(get_examples=lambda: train_examples)
    tag_scores, bp_tag_scores = morph_ls.model.begin_update([eg.predicted for eg in train_examples])
    ops = get_current_ops()
    no_ls_grads = ops.to_numpy(morph_no_ls.get_loss(train_examples, tag_scores)[1][0])
    ls_grads = ops.to_numpy(morph_ls.get_loss(train_examples, tag_scores)[1][0])
    assert_almost_equal(ls_grads / no_ls_grads, 0.94285715)