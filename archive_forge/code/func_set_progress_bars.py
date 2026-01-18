import threading
from typing import Any, List, Optional
import ray
from ray.experimental import tqdm_ray
from ray.types import ObjectRef
from ray.util.annotations import PublicAPI
@PublicAPI
def set_progress_bars(enabled: bool) -> bool:
    """Set whether progress bars are enabled.

    The default behavior is controlled by the
    ``RAY_DATA_DISABLE_PROGRESS_BARS`` environment variable. By default,
    it is set to "0". Setting it to "1" will disable progress bars, unless
    they are reenabled by this method.

    Returns:
        Whether progress bars were previously enabled.
    """
    from ray.data import DataContext
    ctx = DataContext.get_current()
    old_value = ctx.enable_progress_bars
    ctx.enable_progress_bars = enabled
    return old_value