from functools import partial
from typing import List, Optional, Tuple, cast
from thinc.api import (
from thinc.layers.chain import init as init_chain
from thinc.layers.resizable import resize_linear_weighted, resize_model
from thinc.types import ArrayXd, Floats2d
from ...attrs import ORTH
from ...errors import Errors
from ...tokens import Doc
from ...util import registry
from ..extract_ngrams import extract_ngrams
from ..staticvectors import StaticVectors
from .tok2vec import get_tok2vec_width
@registry.architectures('spacy.TextCatCNN.v2')
def build_simple_cnn_text_classifier(tok2vec: Model, exclusive_classes: bool, nO: Optional[int]=None) -> Model[List[Doc], Floats2d]:
    """
    Build a simple CNN text classifier, given a token-to-vector model as inputs.
    If exclusive_classes=True, a softmax non-linearity is applied, so that the
    outputs sum to 1. If exclusive_classes=False, a logistic non-linearity
    is applied instead, so that outputs are in the range [0, 1].
    """
    return build_reduce_text_classifier(tok2vec=tok2vec, exclusive_classes=exclusive_classes, use_reduce_first=False, use_reduce_last=False, use_reduce_max=False, use_reduce_mean=True, nO=nO)