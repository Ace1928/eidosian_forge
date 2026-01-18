import base64
import json
import logging
import re
import uuid
from xml.dom import minidom
from IPython.display import HTML, display
from rdkit import Chem
from rdkit.Chem import Draw
from . import rdMolDraw2D
def setOpt(mol, key, value):
    if not isinstance(mol, Chem.Mol) or not isinstance(key, str) or (not key):
        raise ValueError(f'Bad args ({str(type(mol))}, {str(type(key))}) for {__name__}.setOpt(mol: Chem.Mol, key: str, value: Any)')
    opts = getOpts(mol)
    opts[key] = value
    setOpts(mol, opts)