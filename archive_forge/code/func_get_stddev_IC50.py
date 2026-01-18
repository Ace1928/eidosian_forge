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
def get_stddev_IC50(mol_list):
    IC50_list = []
    for mol in mol_list:
        try:
            IC50_list.append(round(float(mol.GetProp('value')), 2))
        except Exception:
            print('no IC50 reported', mol.GetProp('_Name'))
    IC50_stddev = np.std(IC50_list, ddof=1)
    return (IC50_stddev, IC50_list)