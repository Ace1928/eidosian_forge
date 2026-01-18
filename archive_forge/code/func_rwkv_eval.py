import os
import sys
import ctypes
import pathlib
import platform
from typing import Optional, List, Tuple, Callable
def rwkv_eval(self, ctx: RWKVContext, token: int, state_in_address: Optional[int], state_out_address: int, logits_out_address: int) -> None:
    """
        Evaluates the model for a single token.
        Throws an exception in case of any error. Error messages would be printed to stderr.
        Not thread-safe. For parallel inference, call rwkv_clone_context to create one rwkv_context for each thread.

        Parameters
        ----------
        ctx : RWKVContext
            RWKV context obtained from rwkv_init_from_file.
        token : int
            Next token index, in range 0 <= token < n_vocab.
        state_in_address : int
            Address of the first element of a FP32 buffer of size rwkv_get_state_buffer_element_count; or None, if this is a first pass.
        state_out_address : int
            Address of the first element of a FP32 buffer of size rwkv_get_state_buffer_element_count. This buffer will be written to.
        logits_out_address : int
            Address of the first element of a FP32 buffer of size rwkv_get_logits_buffer_element_count. This buffer will be written to.
        """
    if not self.library.rwkv_eval(ctx.ptr, ctypes.c_int32(token), ctypes.cast(0 if state_in_address is None else state_in_address, P_FLOAT), ctypes.cast(state_out_address, P_FLOAT), ctypes.cast(logits_out_address, P_FLOAT)):
        raise ValueError('rwkv_eval failed, check stderr')