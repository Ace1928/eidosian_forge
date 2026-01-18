import io
import json
import os.path
import pickle
import tempfile
import torch
from torch.utils.data.datapipes.utils.common import StreamWrapper
def decode1(self, key, data):
    if not data:
        return data
    if Decoder._is_stream_handle(data):
        ds = data
        data = b''.join(data)
        ds.close()
    for f in self.handlers:
        result = f(key, data)
        if result is not None:
            return result
    return data