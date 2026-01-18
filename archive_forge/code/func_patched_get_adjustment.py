import importlib
import logging
import re
from io import StringIO
from xml.dom import minidom
from xml.parsers.expat import ExpatError
from rdkit.Chem import Mol
def patched_get_adjustment():
    """ Avoid truncation of data frame values that contain HTML content """
    adj = orig_get_adjustment()
    orig_len = adj.len
    adj.len = lambda text, *args, **kwargs: 0 if is_molecule_image(text) else orig_len(text, *args, **kwargs)
    return adj