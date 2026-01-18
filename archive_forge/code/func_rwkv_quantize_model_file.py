import os
import sys
import ctypes
import pathlib
import platform
from typing import Optional, List, Tuple, Callable
def rwkv_quantize_model_file(self, model_file_path_in: str, model_file_path_out: str, format_name: str) -> None:
    """
        Quantizes FP32 or FP16 model to one of INT4 formats.
        Throws an exception in case of any error. Error messages would be printed to stderr.

        Parameters
        ----------
        model_file_path_in : str
            Path to model file in ggml format, must be either FP32 or FP16.
        model_file_path_out : str
            Quantized model will be written here.
        format_name : str
            One of QUANTIZED_FORMAT_NAMES.
        """
    if format_name not in QUANTIZED_FORMAT_NAMES:
        raise ValueError(f'Unknown format name {format_name}, use one of {QUANTIZED_FORMAT_NAMES}')
    if not self.library.rwkv_quantize_model_file(model_file_path_in.encode('utf-8'), model_file_path_out.encode('utf-8'), format_name.encode('utf-8')):
        raise ValueError('rwkv_quantize_model_file failed, check stderr')