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
@pytest.mark.parametrize('textcat_config', [single_label_default_config, single_label_bow_config, single_label_cnn_config, multi_label_default_config, multi_label_bow_config, multi_label_cnn_config])
@pytest.mark.issue(5551)
def test_issue5551(textcat_config):
    """Test that after fixing the random seed, the results of the pipeline are truly identical"""
    component = 'textcat'
    pipe_cfg = Config().from_str(textcat_config)
    results = []
    for i in range(3):
        fix_random_seed(0)
        nlp = English()
        text = 'Once hot, form ping-pong-ball-sized balls of the mixture, each weighing roughly 25 g.'
        annots = {'cats': {'Labe1': 1.0, 'Label2': 0.0, 'Label3': 0.0}}
        pipe = nlp.add_pipe(component, config=pipe_cfg, last=True)
        for label in set(annots['cats']):
            pipe.add_label(label)
        nlp.initialize()
        doc = nlp.make_doc(text)
        nlp.update([Example.from_dict(doc, annots)])
        result = pipe.model.predict([doc])
        results.append(result[0])
    assert len(results) == 3
    ops = get_current_ops()
    assert_almost_equal(ops.to_numpy(results[0]), ops.to_numpy(results[1]), decimal=5)
    assert_almost_equal(ops.to_numpy(results[0]), ops.to_numpy(results[2]), decimal=5)