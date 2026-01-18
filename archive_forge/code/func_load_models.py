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
def load_models(self, model_files):
    """load model or list of models into self.model"""
    if type(model_files) == str:
        model_files = [model_files]
    i = 0
    for mod_file in model_files:
        model = open(mod_file, 'r')
        unPickled = Unpickler(model)
        clf_RF = unPickled.load()
        self.model.append(clf_RF)
        model.close()
        i += 1
    return i