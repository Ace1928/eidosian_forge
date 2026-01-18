import hashlib
import logging
import os
from io import BytesIO
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Type, Union, cast
from urllib import parse
import wandb
from wandb import util
from wandb.sdk.lib import hashutil, runid
from wandb.sdk.lib.paths import LogicalPath
from ._private import MEDIA_TMP
from .base_types.media import BatchableMedia, Media
from .helper_types.bounding_boxes_2d import BoundingBoxes2D
from .helper_types.classes import Classes
from .helper_types.image_mask import ImageMask
@classmethod
def all_masks(cls: Type['Image'], images: Sequence['Image'], run: 'LocalRun', run_key: str, step: Union[int, str]) -> Union[List[Optional[dict]], bool]:
    all_mask_groups: List[Optional[dict]] = []
    for image in images:
        if image._masks:
            mask_group = {}
            for k in image._masks:
                mask = image._masks[k]
                mask_group[k] = mask.to_json(run)
            all_mask_groups.append(mask_group)
        else:
            all_mask_groups.append(None)
    if all_mask_groups and (not all((x is None for x in all_mask_groups))):
        return all_mask_groups
    else:
        return False