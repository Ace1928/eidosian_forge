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
def step_1_keeplargestfrag(self):
    """remove all smaller Fragments per compound, just keep the largest"""
    result = []
    for cpd in self.sd_entries:
        fragments = Chem.GetMolFrags(cpd, asMols=True)
        list_cpds_fragsize = []
        for frag in fragments:
            list_cpds_fragsize.append(frag.GetNumAtoms())
        largest_frag_index = list_cpds_fragsize.index(max(list_cpds_fragsize))
        largest_frag = fragments[largest_frag_index]
        result.append(largest_frag)
    self.sd_entries = result
    return True