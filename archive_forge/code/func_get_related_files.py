import sys
import pickle
import errno
import subprocess as sp
import gzip
import hashlib
import locale
from hashlib import md5
import os
import os.path as op
import re
import shutil
import contextlib
import posixpath
from pathlib import Path
import simplejson as json
from time import sleep, time
from .. import logging, config, __version__ as version
from .misc import is_container
def get_related_files(filename, include_this_file=True):
    """Returns a list of related files, as defined in
    ``related_filetype_sets``, for a filename. (e.g., Nifti-Pair, Analyze (SPM)
    and AFNI files).

    Parameters
    ----------
    filename : str
        File name to find related filetypes of.
    include_this_file : bool
        If true, output includes the input filename.
    """
    related_files = []
    path, name, this_type = split_filename(filename)
    for type_set in related_filetype_sets:
        if this_type in type_set:
            for related_type in type_set:
                if include_this_file or related_type != this_type:
                    related_files.append(op.join(path, name + related_type))
    if not len(related_files):
        related_files = [filename]
    return related_files