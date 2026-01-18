import argparse
import os
import pickle
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem
def getArgParser():
    """ Create the argument parser """
    parser = argparse.ArgumentParser('Fast clustering for chemoinformatics')
    parser.add_argument('input', help='filename of input file')
    parser.add_argument('nclusters', metavar='N', help='the number of clusters')
    parser.add_argument('--output', help='filename of output, tab separated format', default='clustered.tsv')
    parser.add_argument('--centroid', metavar='CENTROID', help='filename of centroid information. tab separated format', default='centroid.tsv')
    return parser