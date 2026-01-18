import importlib
import logging
import re
from io import StringIO
from xml.dom import minidom
from xml.parsers.expat import ExpatError
from rdkit.Chem import Mol
def changeMoleculeRendering(frame, renderer='image'):
    if not renderer.lower().startswith('str'):
        set_rdk_attr(frame, RDK_MOLS_AS_IMAGE_ATTR)
    elif hasattr(frame, RDK_MOLS_AS_IMAGE_ATTR):
        delattr(frame, RDK_MOLS_AS_IMAGE_ATTR)