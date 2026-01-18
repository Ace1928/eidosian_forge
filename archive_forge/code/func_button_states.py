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
def button_states(self):
    """Return the on/off state names for button widgets.

        A button may have 'normal' or 'pressed down' appearances. While the 'Off'
        state is usually called like this, the 'On' state is often given a name
        relating to the functional context.
        """
    if self.field_type not in (2, 5):
        return None
    if hasattr(self, 'parent'):
        doc = self.parent.parent
    else:
        return
    xref = self.xref
    states = {'normal': None, 'down': None}
    APN = doc.xref_get_key(xref, 'AP/N')
    if APN[0] == 'dict':
        nstates = []
        APN = APN[1][2:-2]
        apnt = APN.split('/')[1:]
        for x in apnt:
            nstates.append(x.split()[0])
        states['normal'] = nstates
    if APN[0] == 'xref':
        nstates = []
        nxref = int(APN[1].split(' ')[0])
        APN = doc.xref_object(nxref)
        apnt = APN.split('/')[1:]
        for x in apnt:
            nstates.append(x.split()[0])
        states['normal'] = nstates
    APD = doc.xref_get_key(xref, 'AP/D')
    if APD[0] == 'dict':
        dstates = []
        APD = APD[1][2:-2]
        apdt = APD.split('/')[1:]
        for x in apdt:
            dstates.append(x.split()[0])
        states['down'] = dstates
    if APD[0] == 'xref':
        dstates = []
        dxref = int(APD[1].split(' ')[0])
        APD = doc.xref_object(dxref)
        apdt = APD.split('/')[1:]
        for x in apdt:
            dstates.append(x.split()[0])
        states['down'] = dstates
    return states