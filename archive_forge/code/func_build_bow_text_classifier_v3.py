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
@registry.architectures('spacy.TextCatBOW.v3')
def build_bow_text_classifier_v3(exclusive_classes: bool, ngram_size: int, no_output_layer: bool, length: int=262144, nO: Optional[int]=None) -> Model[List[Doc], Floats2d]:
    if length < 1:
        raise ValueError(Errors.E1056.format(length=length))
    length = 2 ** (length - 1).bit_length()
    return _build_bow_text_classifier(exclusive_classes=exclusive_classes, ngram_size=ngram_size, no_output_layer=no_output_layer, nO=nO, sparse_linear=SparseLinear_v2(nO=nO, length=length))