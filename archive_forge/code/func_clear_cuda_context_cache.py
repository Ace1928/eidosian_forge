from typing import Dict, List, Tuple
import torio
def clear_cuda_context_cache():
    """Clear the CUDA context used by CUDA Hardware accelerated video decoding"""
    ffmpeg_ext.clear_cuda_context_cache()