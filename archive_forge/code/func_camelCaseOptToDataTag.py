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
def camelCaseOptToDataTag(opt):
    tag = _camelCaseOptToTagRe.sub(_dashLower, opt)
    if not tag.startswith('data-'):
        tag = 'data-' + tag
    return tag