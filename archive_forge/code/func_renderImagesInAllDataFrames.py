import importlib
import logging
import re
from io import StringIO
from xml.dom import minidom
from xml.parsers.expat import ExpatError
from rdkit.Chem import Mol
def renderImagesInAllDataFrames(images=True):
    if images:
        set_rdk_attr(pd.core.frame.DataFrame, RDK_MOLS_AS_IMAGE_ATTR)
    elif hasattr(pd.core.frame.DataFrame, RDK_MOLS_AS_IMAGE_ATTR):
        delattr(pd.core.frame.DataFrame, RDK_MOLS_AS_IMAGE_ATTR)