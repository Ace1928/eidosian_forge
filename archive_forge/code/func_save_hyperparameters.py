import copy
import inspect
import types
from argparse import Namespace
from typing import Any, List, MutableMapping, Optional, Sequence, Union
from lightning_fabric.utilities.data import AttributeDict
from pytorch_lightning.utilities.parsing import save_hyperparameters
def save_hyperparameters(self, *args: Any, ignore: Optional[Union[Sequence[str], str]]=None, frame: Optional[types.FrameType]=None, logger: bool=True) -> None:
    """Save arguments to ``hparams`` attribute.

        Args:
            args: single object of `dict`, `NameSpace` or `OmegaConf`
                or string names or arguments from class ``__init__``
            ignore: an argument name or a list of argument names from
                class ``__init__`` to be ignored
            frame: a frame object. Default is None
            logger: Whether to send the hyperparameters to the logger. Default: True

        Example::
            >>> from pytorch_lightning.core.mixins import HyperparametersMixin
            >>> class ManuallyArgsModel(HyperparametersMixin):
            ...     def __init__(self, arg1, arg2, arg3):
            ...         super().__init__()
            ...         # manually assign arguments
            ...         self.save_hyperparameters('arg1', 'arg3')
            ...     def forward(self, *args, **kwargs):
            ...         ...
            >>> model = ManuallyArgsModel(1, 'abc', 3.14)
            >>> model.hparams
            "arg1": 1
            "arg3": 3.14

            >>> from pytorch_lightning.core.mixins import HyperparametersMixin
            >>> class AutomaticArgsModel(HyperparametersMixin):
            ...     def __init__(self, arg1, arg2, arg3):
            ...         super().__init__()
            ...         # equivalent automatic
            ...         self.save_hyperparameters()
            ...     def forward(self, *args, **kwargs):
            ...         ...
            >>> model = AutomaticArgsModel(1, 'abc', 3.14)
            >>> model.hparams
            "arg1": 1
            "arg2": abc
            "arg3": 3.14

            >>> from pytorch_lightning.core.mixins import HyperparametersMixin
            >>> class SingleArgModel(HyperparametersMixin):
            ...     def __init__(self, params):
            ...         super().__init__()
            ...         # manually assign single argument
            ...         self.save_hyperparameters(params)
            ...     def forward(self, *args, **kwargs):
            ...         ...
            >>> model = SingleArgModel(Namespace(p1=1, p2='abc', p3=3.14))
            >>> model.hparams
            "p1": 1
            "p2": abc
            "p3": 3.14

            >>> from pytorch_lightning.core.mixins import HyperparametersMixin
            >>> class ManuallyArgsModel(HyperparametersMixin):
            ...     def __init__(self, arg1, arg2, arg3):
            ...         super().__init__()
            ...         # pass argument(s) to ignore as a string or in a list
            ...         self.save_hyperparameters(ignore='arg2')
            ...     def forward(self, *args, **kwargs):
            ...         ...
            >>> model = ManuallyArgsModel(1, 'abc', 3.14)
            >>> model.hparams
            "arg1": 1
            "arg3": 3.14

        """
    self._log_hyperparams = logger
    if not frame:
        current_frame = inspect.currentframe()
        if current_frame:
            frame = current_frame.f_back
    save_hyperparameters(self, *args, ignore=ignore, frame=frame)