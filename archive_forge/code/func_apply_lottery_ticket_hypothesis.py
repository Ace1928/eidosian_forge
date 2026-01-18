import inspect
import logging
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch.nn.utils.prune as pytorch_prune
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor, nn
from typing_extensions import TypedDict, override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.core.module import LightningModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_debug, rank_zero_only
def apply_lottery_ticket_hypothesis(self) -> None:
    """Lottery ticket hypothesis algorithm (see page 2 of the paper):

            1. Randomly initialize a neural network :math:`f(x; \\theta_0)` (where :math:`\\theta_0 \\sim \\mathcal{D}_\\theta`).
            2. Train the network for :math:`j` iterations, arriving at parameters :math:`\\theta_j`.
            3. Prune :math:`p\\%` of the parameters in :math:`\\theta_j`, creating a mask :math:`m`.
            4. Reset the remaining parameters to their values in :math:`\\theta_0`, creating the winning ticket :math:`f(x; m \\odot \\theta_0)`.

        This function implements the step 4.

        The ``resample_parameters`` argument can be used to reset the parameters with a new :math:`\\theta_z \\sim \\mathcal{D}_\\theta`

        """
    assert self._original_layers is not None
    for d in self._original_layers.values():
        copy = d['data']
        names = d['names']
        if self._resample_parameters and hasattr(copy, 'reset_parameters') and callable(copy.reset_parameters):
            copy = deepcopy(copy)
            copy.reset_parameters()
        for i, name in names:
            new, _ = self._parameters_to_prune[i]
            self._copy_param(new, copy, name)