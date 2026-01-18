from abc import ABCMeta
import glob
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import warnings
from ray.util.annotations import PublicAPI, DeveloperAPI
from ray.tune.utils.util import _atomic_save, _load_newest_checkpoint
@classmethod
def need_override_by_subclass(mcs, attr_name: str, attr: Any) -> bool:
    return (attr_name.startswith('on_') and (not attr_name.startswith('on_trainer_init')) or attr_name == 'setup') and callable(attr)