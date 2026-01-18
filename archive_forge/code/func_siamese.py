from typing import Callable, Optional, Tuple, TypeVar
from ..config import registry
from ..model import Model
from ..types import ArrayXd
from ..util import get_width
@registry.layers('siamese.v1')
def siamese(layer: Model[LayerT, SimT], similarity: Model[Tuple[SimT, SimT], OutT]) -> Model[InT, OutT]:
    return Model(f'siamese({layer.name}, {similarity.name})', forward, init=init, layers=[layer, similarity], dims={'nI': layer.get_dim('nI'), 'nO': similarity.get_dim('nO')})