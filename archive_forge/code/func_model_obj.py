import os
import shutil
import sys
from typing import (
import wandb
from wandb import util
from wandb.sdk.lib import runid
from wandb.sdk.lib.hashutil import md5_file_hex
from wandb.sdk.lib.paths import LogicalPath
from ._private import MEDIA_TMP
from .base_types.wb_value import WBValue
def model_obj(self) -> SavedModelObjType:
    """Return the model object."""
    if self._model_obj is None:
        assert self._path is not None, 'Cannot load model object without path'
        self._set_obj(self._deserialize(self._path))
    assert self._model_obj is not None, 'Model object is None'
    return self._model_obj