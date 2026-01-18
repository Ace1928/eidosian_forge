import logging
from pathlib import Path
from typing import List, Union
from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer
def maybe_load_preprocessors(src_name_or_path: Union[str, Path], subfolder: str='', trust_remote_code: bool=False) -> List:
    preprocessors = []
    try:
        preprocessors.append(AutoTokenizer.from_pretrained(src_name_or_path, subfolder=subfolder, trust_remote_code=trust_remote_code))
    except Exception:
        pass
    try:
        preprocessors.append(AutoProcessor.from_pretrained(src_name_or_path, subfolder=subfolder, trust_remote_code=trust_remote_code))
    except Exception:
        pass
    try:
        preprocessors.append(AutoFeatureExtractor.from_pretrained(src_name_or_path, subfolder=subfolder, trust_remote_code=trust_remote_code))
    except Exception:
        pass
    return preprocessors