import argparse
from functools import reduce
import logging
import operator
import datasets.wikitext2_data as wikitext2_data
from models import transformer_lm
import numpy as np
import torch
from torch.optim import Adam
def get_dataset_info(args):
    assert args.model_name == 'lm'
    if args.use_synthetic_data:
        return wikitext2_data.get_synthetic_datasets()
    else:
        return wikitext2_data.get_real_datasets()