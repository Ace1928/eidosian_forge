import logging
import sys
from base64 import b64encode
import numpy as np
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, SDWriter, rdchem
from rdkit.Chem.Scaffolds import MurckoScaffold
from io import BytesIO
from xml.dom import minidom
@unittest.skipIf(pd is None, 'pandas not installed')
def testGithub3701(self):
    """ problem with update to pandas v1.2.0 """
    df = pd.DataFrame({'name': ['ethanol', 'furan'], 'smiles': ['CCO', 'c1ccoc1']})
    AddMoleculeColumnToFrame(df, 'smiles', 'molecule')
    self.assertEqual(len(df.molecule), 2)