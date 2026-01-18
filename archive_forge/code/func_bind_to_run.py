import numbers
import os
from typing import TYPE_CHECKING, Optional, Type, Union
import wandb
from wandb import util
from wandb.sdk.lib import runid
from .._private import MEDIA_TMP
from ..base_types.media import Media
def bind_to_run(self, run: 'LocalRun', key: Union[int, str], step: Union[int, str], id_: Optional[Union[int, str]]=None, ignore_copy_err: Optional[bool]=None) -> None:
    super().bind_to_run(run, key, step, id_=id_, ignore_copy_err=ignore_copy_err)
    if hasattr(self, '_val') and 'class_labels' in self._val:
        class_labels = self._val['class_labels']
        run._add_singleton('mask/class_labels', str(key) + '_wandb_delimeter_' + self._key, class_labels)