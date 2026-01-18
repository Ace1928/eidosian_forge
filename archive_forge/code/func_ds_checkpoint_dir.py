from __future__ import annotations
import os
from typing import Any
import torch
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning.strategies.deepspeed import _DEEPSPEED_AVAILABLE
def ds_checkpoint_dir(checkpoint_dir: _PATH, tag: str | None=None) -> str:
    if tag is None:
        latest_path = os.path.join(checkpoint_dir, 'latest')
        if os.path.isfile(latest_path):
            with open(latest_path) as fd:
                tag = fd.read().strip()
        else:
            raise ValueError(f"Unable to find 'latest' file at {latest_path}")
    directory = os.path.join(checkpoint_dir, tag)
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory '{ds_checkpoint_dir}' doesn't exist")
    return directory