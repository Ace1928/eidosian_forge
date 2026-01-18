import gzip
import math
import os.path
import pickle
import sys
from collections import namedtuple
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
def readNPModel(filename=os.path.join(os.path.dirname(__file__), 'publicnp.model.gz')):
    """Reads and returns the scoring model,
  which has to be passed to the scoring functions."""
    print('reading NP model ...', file=sys.stderr)
    fscore = pickle.load(gzip.open(filename))
    print('model in', file=sys.stderr)
    return fscore