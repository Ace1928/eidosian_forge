import io
import json
import os.path
import pickle
import tempfile
import torch
from torch.utils.data.datapipes.utils.common import StreamWrapper
def imagehandler(imagespec):
    return ImageHandler(imagespec)