from typing import Callable, List, Tuple
from thinc.api import Model, chain, with_array
from thinc.types import Floats1d, Floats2d
from ...tokens import Doc
from ...util import registry
@registry.architectures('spacy.SpanFinder.v1')
def build_finder_model(tok2vec: Model[InT, List[Floats2d]], scorer: Model[OutT, OutT]) -> Model[InT, OutT]:
    logistic_layer: Model[List[Floats2d], List[Floats2d]] = with_array(scorer)
    model: Model[InT, OutT] = chain(tok2vec, logistic_layer, flattener())
    model.set_ref('tok2vec', tok2vec)
    model.set_ref('scorer', scorer)
    model.set_ref('logistic_layer', logistic_layer)
    return model