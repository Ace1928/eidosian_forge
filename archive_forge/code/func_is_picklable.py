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
def is_picklable(obj: object) -> bool:
    """Tests if an object can be pickled."""
    try:
        pickle.dumps(obj)
        return True
    except (pickle.PickleError, AttributeError, RuntimeError, TypeError):
        return False