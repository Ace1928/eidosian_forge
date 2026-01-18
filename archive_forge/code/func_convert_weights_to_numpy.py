import logging
import json
import numpy as np
from mxnet import ndarray as nd
@staticmethod
def convert_weights_to_numpy(weights_dict):
    """Convert weights to numpy"""
    return dict([(k.replace('arg:', '').replace('aux:', ''), v.asnumpy()) for k, v in weights_dict.items()])