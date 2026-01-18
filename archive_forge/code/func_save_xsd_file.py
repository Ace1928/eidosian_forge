import os
import warnings
from collections import Counter
from xml.parsers import expat
from io import BytesIO
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
from urllib.request import urlopen
from urllib.parse import urlparse
from Bio import StreamModeError
def save_xsd_file(self, filename, text):
    """Save XSD file to cache."""
    if DataHandler.local_xsd_dir is None:
        return
    path = os.path.join(DataHandler.local_xsd_dir, filename)
    try:
        handle = open(path, 'wb')
    except OSError:
        warnings.warn(f'Failed to save {filename} at {path}')
    else:
        handle.write(text)
        handle.close()