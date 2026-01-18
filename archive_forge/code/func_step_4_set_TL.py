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
def step_4_set_TL(self, threshold, ic50_tag='value'):
    """set Property "TL"(TrafficLight) for each compound:
        if ic50_tag (default:"value") > threshold: TL = 0, else 1"""
    result = []
    i, j = (0, 0)
    for cpd in self.sd_entries:
        if float(cpd.GetProp(ic50_tag)) > float(threshold):
            cpd.SetProp('TL', '0')
            i += 1
        else:
            cpd.SetProp('TL', '1')
            j += 1
        result.append(cpd)
    self.sd_entries = result
    if self.verbous:
        print('## act: %d, inact: %d' % (j, i))
    return True