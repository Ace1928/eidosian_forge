import io
import json
import os.path
import pickle
import tempfile
import torch
from torch.utils.data.datapipes.utils.common import StreamWrapper
def audiohandler(extension, data):
    if extension not in ['flac', 'mp3', 'sox', 'wav', 'm4a', 'ogg', 'wma']:
        return None
    try:
        import torchaudio
    except ImportError as e:
        raise ModuleNotFoundError('Package `torchaudio` is required to be installed for default audio file loader.Please use `pip install torchaudio` or `conda install torchaudio -c pytorch`to install the package') from e
    with tempfile.TemporaryDirectory() as dirname:
        fname = os.path.join(dirname, f'file.{extension}')
        with open(fname, 'wb') as stream:
            stream.write(data)
            return torchaudio.load(fname)