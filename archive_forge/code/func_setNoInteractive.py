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
def setNoInteractive(mol, shouldSet=True):
    opts = getOpts(mol)
    if shouldSet:
        opts[_disabled] = True
    elif _disabled in opts:
        opts.pop(_disabled)
    setOpts(mol, opts)