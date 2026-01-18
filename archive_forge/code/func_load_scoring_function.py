import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
import sys
import os
from importlib import import_module
from .tpot import TPOTClassifier, TPOTRegressor
from ._version import __version__
def load_scoring_function(scoring_func):
    """
    converts mymodule.myfunc in the myfunc
    object itself so tpot receives a scoring function
    """
    if scoring_func and '.' in scoring_func:
        try:
            module_name, func_name = scoring_func.rsplit('.', 1)
            module_path = os.getcwd()
            sys.path.insert(0, module_path)
            scoring_func = getattr(import_module(module_name), func_name)
            sys.path.pop(0)
            print('manual scoring function: {}'.format(scoring_func))
            print('taken from module: {}'.format(module_name))
        except Exception as e:
            print('failed importing custom scoring function, error: {}'.format(str(e)))
            raise ValueError(e)
    return scoring_func