import io
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import wandb
import wandb.util
from wandb.sdk.lib import telemetry
from wandb.viz import custom_chart
def encode_images(_img_strs: List[bytes], _value: Any) -> None:
    try:
        from PIL import Image
    except ImportError:
        wandb.termwarn('Install pillow if you are logging images with Tensorboard. To install, run `pip install pillow`.', repeat=False)
        return None
    if len(_img_strs) == 0:
        return None
    images: List[Union[wandb.Video, wandb.Image]] = []
    for _img_str in _img_strs:
        if _img_str.startswith(b'GIF'):
            images.append(wandb.Video(io.BytesIO(_img_str), format='gif'))
        else:
            images.append(wandb.Image(Image.open(io.BytesIO(_img_str))))
    tag_idx = _value.tag.rsplit('/', 1)
    if len(tag_idx) > 1 and tag_idx[1].isdigit():
        tag, idx = tag_idx
        values.setdefault(history_image_key(tag, namespace), []).extend(images)
    else:
        values[history_image_key(_value.tag, namespace)] = images
    return None