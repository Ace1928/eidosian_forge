import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
def embfile_info(self, item: typing.Union[int, str]) -> dict:
    """Get information of an item in the EmbeddedFiles array.

        Args:
            item: number or name of item.
        Returns:
            Information dictionary.
        """
    idx = self._embeddedFileIndex(item)
    infodict = {'name': self.embfile_names()[idx]}
    xref = self._embfile_info(idx, infodict)
    t, date = self.xref_get_key(xref, 'Params/CreationDate')
    if t != 'null':
        infodict['creationDate'] = date
    t, date = self.xref_get_key(xref, 'Params/ModDate')
    if t != 'null':
        infodict['modDate'] = date
    t, md5 = self.xref_get_key(xref, 'Params/CheckSum')
    if t != 'null':
        infodict['checksum'] = binascii.hexlify(md5.encode()).decode()
    return infodict