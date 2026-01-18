import csv
import gzip
import json
import math
import optparse
import os
import pickle
import re
import sys
from pickle import Unpickler
import numpy as np
import requests
from pylab import *
from scipy import interp, stats
from sklearn import cross_validation, metrics, preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (auc, make_scorer, precision_score, recall_score,
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, SDWriter
from rdkit.ML.Descriptors import MoleculeDescriptors
from the one dimensional weights.
def int_ifclose(x, dec=1, width=4):
    """helper function for creating result string for int or float

only dec=1 and width=4 is implemented

Parameters
----------
x : int or float
value to format
dec : 1
number of decimals to print if x is not an integer
width : 4
width of string

Returns
-------
xint : int or float
x is converted to int if it is within 1e-14 of an integer
x_string : str
x formatted as string, either '%4d' or '%4.1f'
"""
    xint = int(round(x))
    if np.max(np.abs(xint - x)) < 1e-14:
        return (xint, '%4d' % xint)
    else:
        return (x, '%4.1f' % x)