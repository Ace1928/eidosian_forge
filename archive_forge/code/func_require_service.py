from typing import Optional, Sequence, Union
import wandb
from wandb.errors import UnsupportedError
from wandb.sdk import wandb_run
from wandb.sdk.lib.wburls import wburls
def require_service(self) -> None:
    self._require_service()