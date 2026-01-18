import pickle
import sys
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
def distFunc(a, b):
    return 1.0 - DataStructs.DiceSimilarity(a[0], b[0])