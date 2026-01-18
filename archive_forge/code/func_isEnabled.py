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
def isEnabled(mol=None):
    return _enabled_div_uuid and (mol is None or not isNoInteractive(mol))