import os
import argparse
from convert_model import convert_model
from convert_mean import convert_mean
import mxnet as mx
def get_model_meta_info(model_name):
    """returns a dict with model information"""
    return model_meta_info[model_name].copy()