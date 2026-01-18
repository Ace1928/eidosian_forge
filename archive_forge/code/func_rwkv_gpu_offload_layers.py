import os
import sys
import ctypes
import pathlib
import platform
from typing import Optional, List, Tuple, Callable
def rwkv_gpu_offload_layers(self, ctx: RWKVContext, layer_count: int) -> bool:
    """
        Offloads specified count of model layers onto the GPU. Offloaded layers are evaluated using cuBLAS or CLBlast.
        For the purposes of this function, model head (unembedding matrix) is treated as an additional layer:
        - pass `rwkv_get_n_layer(ctx)` to offload all layers except model head
        - pass `rwkv_get_n_layer(ctx) + 1` to offload all layers, including model head
        Returns true if at least one layer was offloaded.
        If rwkv.cpp was compiled without cuBLAS and CLBlast support, this function is a no-op and always returns false.

        Parameters
        ----------
        ctx : RWKVContext
            RWKV context obtained from rwkv_init_from_file.
        layer_count : int
            Count of layers to offload onto the GPU, must be >= 0.
        """
    if not layer_count >= 0:
        raise ValueError('Layer count must be >= 0')
    return self.library.rwkv_gpu_offload_layers(ctx.ptr, ctypes.c_uint32(layer_count))