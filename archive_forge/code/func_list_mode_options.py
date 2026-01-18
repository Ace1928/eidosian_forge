from typing import Any, Dict, List, Optional
import torch.fx
import torch.utils._pytree as pytree
def list_mode_options(mode: Optional[str]=None, dynamic: Optional[bool]=None) -> Dict[str, Any]:
    """Returns a dictionary describing the optimizations that each of the available
    modes passed to `torch.compile()` performs.

    Args:
        mode (str, optional): The mode to return the optimizations for.
        If None, returns optimizations for all modes
        dynamic (bool, optional): Whether dynamic shape is enabled.

    Example::
        >>> torch._inductor.list_mode_options()
    """
    mode_options: Dict[str, Dict[str, bool]] = {'default': {}, 'reduce-overhead': {'triton.cudagraphs': True}, 'max-autotune-no-cudagraphs': {'max_autotune': True}, 'max-autotune': {'max_autotune': True, 'triton.cudagraphs': True}}
    return mode_options[mode] if mode else mode_options