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
@registry.architectures('spacy.TextCatReduce.v1')
def build_reduce_text_classifier(tok2vec: Model, exclusive_classes: bool, use_reduce_first: bool, use_reduce_last: bool, use_reduce_max: bool, use_reduce_mean: bool, nO: Optional[int]=None) -> Model[List[Doc], Floats2d]:
    """Build a model that classifies pooled `Doc` representations.

    Pooling is performed using reductions. Reductions are concatenated when
    multiple reductions are used.

    tok2vec (Model): the tok2vec layer to pool over.
    exclusive_classes (bool): Whether or not classes are mutually exclusive.
    use_reduce_first (bool): Pool by using the hidden representation of the
        first token of a `Doc`.
    use_reduce_last (bool): Pool by using the hidden representation of the
        last token of a `Doc`.
    use_reduce_max (bool): Pool by taking the maximum values of the hidden
        representations of a `Doc`.
    use_reduce_mean (bool): Pool by taking the mean of all hidden
        representations of a `Doc`.
    nO (Optional[int]): Number of classes.
    """
    fill_defaults = {'b': 0, 'W': 0}
    reductions = []
    if use_reduce_first:
        reductions.append(reduce_first())
    if use_reduce_last:
        reductions.append(reduce_last())
    if use_reduce_max:
        reductions.append(reduce_max())
    if use_reduce_mean:
        reductions.append(reduce_mean())
    if not len(reductions):
        raise ValueError(Errors.E1057)
    with Model.define_operators({'>>': chain}):
        cnn = tok2vec >> list2ragged() >> concatenate(*reductions)
        nO_tok2vec = tok2vec.maybe_get_dim('nO')
        nI = nO_tok2vec * len(reductions) if nO_tok2vec is not None else None
        if exclusive_classes:
            output_layer = Softmax(nO=nO, nI=nI)
            fill_defaults['b'] = NEG_VALUE
            resizable_layer: Model = resizable(output_layer, resize_layer=partial(resize_linear_weighted, fill_defaults=fill_defaults))
            model = cnn >> resizable_layer
        else:
            output_layer = Linear(nO=nO, nI=nI)
            resizable_layer = resizable(output_layer, resize_layer=partial(resize_linear_weighted, fill_defaults=fill_defaults))
            model = cnn >> resizable_layer >> Logistic()
        model.set_ref('output_layer', output_layer)
        model.attrs['resize_output'] = partial(resize_and_set_ref, resizable_layer=resizable_layer)
    model.set_ref('tok2vec', tok2vec)
    if nO is not None:
        model.set_dim('nO', cast(int, nO))
    model.attrs['multi_label'] = not exclusive_classes
    return model