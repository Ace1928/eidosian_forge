import math
import numpy
from rdkit import Chem, Geometry
def findNeighbors(atomId, adjMat):
    """
  Find the IDs of the neighboring atom IDs
  
  ARGUMENTS:
  atomId - atom of interest
  adjMat - adjacency matrix for the compound
  """
    return [i for i, nid in enumerate(adjMat[atomId]) if nid >= 1]