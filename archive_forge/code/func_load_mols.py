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
def load_mols(self, sd_file):
    """load SD-File from .sdf, .sdf.gz or .sd.gz"""
    if sd_file.endswith('.sdf.gz') or sd_file.endswith('.sd.gz'):
        SDFile = Chem.ForwardSDMolSupplier(gzip.open(sd_file))
    else:
        SDFile = Chem.SDMolSupplier(sd_file)
    self.sd_entries = [mol for mol in SDFile]
    return True