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
def clearOpts(mol):
    if not isinstance(mol, Chem.Mol):
        raise ValueError(f'Bad args ({str(type(mol))}) for {__name__}.clearOpts(mol: Chem.Mol)')
    setOpts(mol, {})