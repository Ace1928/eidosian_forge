import importlib
import logging
import re
from io import StringIO
from xml.dom import minidom
from xml.parsers.expat import ExpatError
from rdkit.Chem import Mol
def check_rdk_attr(frame, attr):
    return hasattr(frame, attr) and getattr(frame, attr)