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
def clean_namespace(hparams: MutableMapping) -> None:
    """Removes all unpicklable entries from hparams."""
    del_attrs = [k for k, v in hparams.items() if not is_picklable(v)]
    for k in del_attrs:
        rank_zero_warn(f"attribute '{k}' removed from hparams because it cannot be pickled")
        del hparams[k]