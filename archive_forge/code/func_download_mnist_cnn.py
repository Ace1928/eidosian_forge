import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import PopulationBasedTraining
import argparse
import os
from filelock import FileLock
import tempfile
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import numpy as np
from ray.tune.examples.pbt_dcgan_mnist.common import (
def download_mnist_cnn():
    import urllib.request
    if not os.path.exists(MODEL_PATH):
        print('downloading model')
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve('https://github.com/ray-project/ray/raw/master/python/ray/tune/examples/pbt_dcgan_mnist/mnist_cnn.pt', MODEL_PATH)
    return MODEL_PATH