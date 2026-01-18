import time
import numpy as np
import torch
import ray
import ray.train.torch
from ray import train, tune
from ray.train import ScalingConfig
from ray.train.horovod import HorovodTrainer
from ray.tune.tune_config import TuneConfig
from ray.tune.tuner import Tuner
def qu(x):
    m3 = 10.0
    m2 = 5.0
    m1 = -20.0
    m0 = -5.0
    return m3 * x * x * x + m2 * x * x + m1 * x + m0