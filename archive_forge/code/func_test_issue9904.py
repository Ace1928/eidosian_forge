import random
import numpy.random
import pytest
from numpy.testing import assert_almost_equal
from thinc.api import Config, compounding, fix_random_seed, get_current_ops
from wasabi import msg
import spacy
from spacy import util
from spacy.cli.evaluate import print_prf_per_type, print_textcats_auc_per_cat
from spacy.lang.en import English
from spacy.language import Language
from spacy.pipeline import TextCategorizer
from spacy.pipeline.textcat import (
from spacy.pipeline.textcat_multilabel import (
from spacy.pipeline.tok2vec import DEFAULT_TOK2VEC_MODEL
from spacy.scorer import Scorer
from spacy.tokens import Doc, DocBin
from spacy.training import Example
from spacy.training.initialize import init_nlp
from ..tok2vec import build_lazy_init_tok2vec as _
from ..util import make_tempdir
@pytest.mark.issue(9904)
def test_issue9904():
    nlp = Language()
    textcat = nlp.add_pipe('textcat')
    get_examples = make_get_examples_single_label(nlp)
    nlp.initialize(get_examples)
    examples = get_examples()
    scores = textcat.predict([eg.predicted for eg in examples])
    loss = textcat.get_loss(examples, scores)[0]
    loss_double_bs = textcat.get_loss(examples * 2, scores.repeat(2, axis=0))[0]
    assert loss == pytest.approx(loss_double_bs)