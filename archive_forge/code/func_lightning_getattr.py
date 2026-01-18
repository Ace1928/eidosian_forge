import copy
import inspect
import pickle
import types
from dataclasses import fields, is_dataclass
from typing import Any, Dict, List, Literal, MutableMapping, Optional, Sequence, Tuple, Type, Union
from torch import nn
import pytorch_lightning as pl
from lightning_fabric.utilities.data import AttributeDict as _AttributeDict
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
def lightning_getattr(model: 'pl.LightningModule', attribute: str) -> Optional[Any]:
    """Special getattr for Lightning. Checks for attribute in model namespace, the old hparams namespace/dict, and the
    datamodule.

    Raises:
        AttributeError:
            If ``model`` doesn't have ``attribute`` in any of
            model namespace, the hparams namespace/dict, and the datamodule.

    """
    holder = _lightning_get_first_attr_holder(model, attribute)
    if holder is None:
        raise AttributeError(f'{attribute} is neither stored in the model namespace nor the `hparams` namespace/dict, nor the datamodule.')
    if isinstance(holder, dict):
        return holder[attribute]
    return getattr(holder, attribute)