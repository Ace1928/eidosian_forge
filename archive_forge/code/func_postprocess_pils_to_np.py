import inspect
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import wandb
from wandb.util import get_module
def postprocess_pils_to_np(image: List) -> 'np_array':
    np = get_module('numpy', required='Please ensure NumPy is installed. You can run `pip install numpy` to install it.')
    return np.stack([np.transpose(np.array(img).astype('uint8'), axes=(2, 0, 1)) for img in image], axis=0)