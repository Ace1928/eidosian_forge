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
@registry.architectures('spacy.TextCatLowData.v1')
def build_text_classifier_lowdata(width: int, dropout: Optional[float], nO: Optional[int]=None) -> Model[List[Doc], Floats2d]:
    with Model.define_operators({'>>': chain, '**': clone}):
        model = StaticVectors(width) >> list2ragged() >> ParametricAttention(width) >> reduce_sum() >> residual(Relu(width, width)) ** 2 >> Linear(nO, width)
        if dropout:
            model = model >> Dropout(dropout)
        model = model >> Logistic()
    return model