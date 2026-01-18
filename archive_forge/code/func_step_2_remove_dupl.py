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
def step_2_remove_dupl(self):
    """remove duplicates from self.sd_entries"""
    result = []
    all_struct_dict = {}
    for cpd in self.sd_entries:
        Chem.RemoveHs(cpd)
        cansmi = Chem.MolToSmiles(cpd, canonical=True)
        if cansmi not in all_struct_dict.keys():
            all_struct_dict[cansmi] = []
        all_struct_dict[cansmi].append(cpd)
    for entry in all_struct_dict.keys():
        if len(all_struct_dict[entry]) == 1:
            all_struct_dict[entry][0].SetProp('cansmirdkit', entry)
            result.append(all_struct_dict[entry][0])
    self.sd_entries = result
    return True