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
@pytest.mark.parametrize('name,get_examples', [('textcat', make_get_examples_single_label), ('textcat_multilabel', make_get_examples_multi_label)])
def test_invalid_label_value(name, get_examples):
    nlp = Language()
    textcat = nlp.add_pipe(name)
    example_getter = get_examples(nlp)

    def invalid_examples():
        examples = example_getter()
        ref = examples[0].reference
        key = list(ref.cats.keys())[0]
        ref.cats[key] = 2.0
        return examples
    with pytest.raises(ValueError):
        nlp.initialize(get_examples=invalid_examples)