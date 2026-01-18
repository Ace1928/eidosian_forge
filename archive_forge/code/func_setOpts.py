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
def setOpts(mol, opts):
    if not isinstance(mol, Chem.Mol) or not isinstance(opts, dict):
        raise ValueError(f'Bad args ({(str(type(mol)), str(type(opts)))}) for {__name__}.setOpts(mol: Chem.Mol, opts: dict)')
    if not all(opts.keys()):
        raise ValueError(f'{__name__}.setOpts(mol: Chem.Mol, opts: dict): no key in opts should be null')
    if opts:
        setattr(mol, _opts, opts)
    elif hasattr(mol, _opts):
        delattr(mol, _opts)