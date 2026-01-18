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
@property
def log_dir(self) -> Optional[str]:
    """Gets the save directory.

        Returns:
            The path to the save directory.

        """
    return self.save_dir