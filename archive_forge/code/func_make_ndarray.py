import io
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import wandb
import wandb.util
from wandb.sdk.lib import telemetry
from wandb.viz import custom_chart
def make_ndarray(tensor: Any) -> Optional['np.ndarray']:
    if tensor_util:
        res = tensor_util.make_ndarray(tensor)
        if res.dtype == 'object':
            return None
        else:
            return res
    else:
        wandb.termwarn("Can't convert tensor summary, upgrade tensorboard with `pip install tensorboard --upgrade`")
        return None