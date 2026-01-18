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
def lightning_setattr(model: 'pl.LightningModule', attribute: str, value: Any) -> None:
    """Special setattr for Lightning. Checks for attribute in model namespace and the old hparams namespace/dict. Will
    also set the attribute on datamodule, if it exists.

    Raises:
        AttributeError:
            If ``model`` doesn't have ``attribute`` in any of
            model namespace, the hparams namespace/dict, and the datamodule.

    """
    holders = _lightning_get_all_attr_holders(model, attribute)
    if len(holders) == 0:
        raise AttributeError(f'{attribute} is neither stored in the model namespace nor the `hparams` namespace/dict, nor the datamodule.')
    for holder in holders:
        if isinstance(holder, dict):
            holder[attribute] = value
        else:
            setattr(holder, attribute, value)