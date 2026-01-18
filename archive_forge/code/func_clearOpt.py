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
def clearOpt(mol, key):
    if not isinstance(mol, Chem.Mol) or not isinstance(key, str):
        raise ValueError(f'Bad args ({str(type(mol))}, {str(type(key))}) for {__name__}.clearOpt(mol: Chem.Mol, key: str)')
    opts = getOpts(mol)
    if key in opts:
        opts.pop(key)
    setOpts(mol, opts)