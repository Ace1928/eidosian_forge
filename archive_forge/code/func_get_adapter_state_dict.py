import inspect
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from ..utils import (
def get_adapter_state_dict(self, adapter_name: Optional[str]=None) -> dict:
    """
        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft

        Gets the adapter state dict that should only contain the weights tensors of the specified adapter_name adapter.
        If no adapter_name is passed, the active adapter is used.

        Args:
            adapter_name (`str`, *optional*):
                The name of the adapter to get the state dict from. If no name is passed, the active adapter is used.
        """
    check_peft_version(min_version=MIN_PEFT_VERSION)
    if not self._hf_peft_config_loaded:
        raise ValueError('No adapter loaded. Please load an adapter first.')
    from peft import get_peft_model_state_dict
    if adapter_name is None:
        adapter_name = self.active_adapter()
    adapter_state_dict = get_peft_model_state_dict(self, adapter_name=adapter_name)
    return adapter_state_dict