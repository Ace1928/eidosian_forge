import io
import json
import os.path
import pickle
import tempfile
import torch
from torch.utils.data.datapipes.utils.common import StreamWrapper
def basichandlers(extension, data):
    if extension in 'txt text transcript':
        return data.decode('utf-8')
    if extension in 'cls cls2 class count index inx id'.split():
        try:
            return int(data)
        except ValueError:
            return None
    if extension in 'json jsn':
        return json.loads(data)
    if extension in 'pyd pickle'.split():
        return pickle.loads(data)
    if extension in 'pt'.split():
        stream = io.BytesIO(data)
        return torch.load(stream)
    return None