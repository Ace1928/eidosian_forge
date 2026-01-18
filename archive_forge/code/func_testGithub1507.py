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
@unittest.skipIf(xlsxwriter is None or pd is None, 'pandas/xlsxwriter not installed')
def testGithub1507(self):
    import os
    from rdkit import RDConfig
    sdfFile = os.path.join(RDConfig.RDDataDir, 'NCI/first_200.props.sdf')
    frame = LoadSDF(sdfFile)
    SaveXlsxFromFrame(frame, 'foo.xlsx', formats={'write_string': {'text_wrap': True}})