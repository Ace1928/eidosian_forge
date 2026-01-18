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
def get_oc_items(self) -> list:
    """Get OCGs and OCMDs used in the page's contents.

        Returns:
            List of items (name, xref, type), where type is one of "ocg" / "ocmd",
            and name is the property name.
        """
    rc = []
    for pname, xref in self._get_resource_properties():
        text = self.parent.xref_object(xref, compressed=True)
        if '/Type/OCG' in text:
            octype = 'ocg'
        elif '/Type/OCMD' in text:
            octype = 'ocmd'
        else:
            continue
        rc.append((pname, xref, octype))
    return rc