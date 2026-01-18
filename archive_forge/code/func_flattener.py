from typing import Callable, List, Tuple
from thinc.api import Model, chain, with_array
from thinc.types import Floats1d, Floats2d
from ...tokens import Doc
from ...util import registry
def flattener() -> Model[List[Floats2d], Floats2d]:
    """Flattens the input to a 1-dimensional list of scores"""

    def forward(model: Model[Floats1d, Floats1d], X: List[Floats2d], is_train: bool) -> Tuple[Floats2d, Callable[[Floats2d], List[Floats2d]]]:
        lens = model.ops.asarray1i([len(doc) for doc in X])
        Y = model.ops.flatten(X)

        def backprop(dY: Floats2d) -> List[Floats2d]:
            return model.ops.unflatten(dY, lens)
        return (Y, backprop)
    return Model('Flattener', forward=forward)