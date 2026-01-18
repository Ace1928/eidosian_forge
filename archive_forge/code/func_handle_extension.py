import io
import json
import os.path
import pickle
import tempfile
import torch
from torch.utils.data.datapipes.utils.common import StreamWrapper
def handle_extension(extensions, f):
    """
    Return a decoder handler function for the list of extensions.

    Extensions can be a space separated list of extensions.
    Extensions can contain dots, in which case the corresponding number
    of extension components must be present in the key given to f.
    Comparisons are case insensitive.
    Examples:
    handle_extension("jpg jpeg", my_decode_jpg)  # invoked for any file.jpg
    handle_extension("seg.jpg", special_case_jpg)  # invoked only for file.seg.jpg
    """
    extensions = extensions.lower().split()

    def g(key, data):
        extension = key.lower().split('.')
        for target in extensions:
            target = target.split('.')
            if len(target) > len(extension):
                continue
            if extension[-len(target):] == target:
                return f(data)
            return None
    return g