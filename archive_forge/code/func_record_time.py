import json, time, random, os
import numpy as np
import torch
from torch.nn import functional as F
def record_time(name):
    if name not in time_slot:
        time_slot[name] = 1e+20
    tt = (time.time_ns() - time_ref) / 1000000000.0
    if tt < time_slot[name]:
        time_slot[name] = tt