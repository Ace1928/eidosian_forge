import os
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Mapping, Optional, Union
from packaging import version
from typing_extensions import override
import wandb
from wandb import Artifact
from wandb.sdk.lib import RunDisabled, telemetry
from wandb.sdk.wandb_run import Run
@override
@rank_zero_only
def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
    params = _convert_params(params)
    params = _sanitize_callable_params(params)
    self.experiment.config.update(params, allow_val_change=True)