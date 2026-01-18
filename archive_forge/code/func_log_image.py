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
@rank_zero_only
def log_image(self, key: str, images: List[Any], step: Optional[int]=None, **kwargs: Any) -> None:
    """Log images (tensors, numpy arrays, PIL Images or file paths).

        Optional kwargs are lists passed to each image (ex: caption, masks, boxes).

        """
    if not isinstance(images, list):
        raise TypeError(f'Expected a list as "images", found {type(images)}')
    n = len(images)
    for k, v in kwargs.items():
        if len(v) != n:
            raise ValueError(f'Expected {n} items but only found {len(v)} for {k}')
    kwarg_list = [{k: kwargs[k][i] for k in kwargs} for i in range(n)]
    metrics = {key: [wandb.Image(img, **kwarg) for img, kwarg in zip(images, kwarg_list)]}
    self.log_metrics(metrics, step)