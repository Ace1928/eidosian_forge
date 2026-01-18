from typing import Any, Dict, Optional, Union
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
def get_standard_metrics(trainer: 'pl.Trainer') -> Dict[str, Union[int, str]]:
    """Returns the standard metrics displayed in the progress bar. Currently, it only includes the version of the
    experiment when using a logger.

    .. code-block::

        Epoch 1:   4%|▎         | 40/1095 [00:03<01:37, 10.84it/s, v_num=10]

    Return:
        Dictionary with the standard metrics to be displayed in the progress bar.

    """
    items_dict: Dict[str, Union[int, str]] = {}
    if trainer.loggers:
        from pytorch_lightning.loggers.utilities import _version
        if (version := _version(trainer.loggers)) not in ('', None):
            if isinstance(version, str):
                version = version[-4:]
            items_dict['v_num'] = version
    return items_dict