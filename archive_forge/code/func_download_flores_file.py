import os
import re
import tempfile
from functools import partial
from typing import Any, ClassVar, Dict, Optional, Sequence, Type
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.functional.text.bleu import _bleu_score_compute, _bleu_score_update
from torchmetrics.utilities.imports import (
@staticmethod
def download_flores_file(model_name: Literal['flores101', 'flores200']) -> None:
    """Download necessary files for `flores` tokenization via `sentencepiece`."""
    import ssl
    import urllib.request
    os.makedirs(_FLORES_LOCAL_DIR, exist_ok=True)
    model_url = _FLORES_MODELS_URL[model_name]
    file_path = os.path.join(_FLORES_LOCAL_DIR, model_url.split('/')[-1])
    try:
        with open(file_path, 'wb') as out_file, urllib.request.urlopen(model_url) as remote_file:
            out_file.write(remote_file.read())
    except ssl.SSLError as e:
        raise OSError(f'Failed to download {model_name} model.') from e