from typing import Type
from torch import optim
from .functional_adadelta import _FunctionalAdadelta
from .functional_adagrad import _FunctionalAdagrad
from .functional_adam import _FunctionalAdam
from .functional_adamax import _FunctionalAdamax
from .functional_adamw import _FunctionalAdamW
from .functional_rmsprop import _FunctionalRMSprop
from .functional_rprop import _FunctionalRprop
from .functional_sgd import _FunctionalSGD
def as_functional_optim(optim_cls: Type, *args, **kwargs):
    try:
        functional_cls = functional_optim_map[optim_cls]
    except KeyError as e:
        raise ValueError(f'Optimizer {optim_cls} does not have a functional counterpart!') from e
    return _create_functional_optim(functional_cls, *args, **kwargs)