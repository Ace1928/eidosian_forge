from json import load
from os.path import join
from typing import Mapping, Optional, Tuple
from huggingface_hub import snapshot_download
def get_model_onnx(model_name: str, device: str, dtype: str, token: Optional[str]=None):
    from uform.onnx_models import VLM_ONNX
    from uform.numpy_preprocessor import NumPyProcessor
    assert device in ('cpu', 'gpu'), f'Invalid `device`: {device}. Must be either `cpu` or `gpu`'
    assert dtype in ('fp32', 'fp16'), f'Invalid `dtype`: {dtype}. Must be either `fp32` or `fp16` (only for gpu)'
    assert device == 'cpu' and dtype == 'fp32' or device == 'gpu', 'Combination `device`=`cpu` & `dtype=fp16` is not supported'
    model_path = snapshot_download(repo_id=f'{model_name}-{device}-{dtype}', token=token)
    with open(join(model_path, 'config.json')) as f:
        config = load(f)
    model = VLM_ONNX(model_path, config, device, dtype)
    processor = NumPyProcessor(config, join(model_path, 'tokenizer.json'))
    return (model, processor)