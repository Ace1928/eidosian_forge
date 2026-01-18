from __future__ import annotations
import abc
import collections
import hashlib
import logging
import os
import shutil
import sys
import tempfile
import traceback
from collections import defaultdict, namedtuple
from typing import TYPE_CHECKING
from xml.etree import ElementTree as Et
import numpy as np
from monty.collections import AttrDict, Namespace
from monty.functools import lazy_property
from monty.itertools import iterator_from_slice
from monty.json import MontyDecoder, MSONable
from monty.os.path import find_exts
from tabulate import tabulate
from pymatgen.core import Element
from pymatgen.core.xcfunc import XcFunc
from pymatgen.io.core import ParseError
from pymatgen.util.plotting import add_fig_kwargs, get_ax_fig
@classmethod
def from_dir(cls, top, exts=None, exclude_dirs='_*') -> Self | None:
    """
        Find all pseudos in the directory tree starting from top.

        Args:
            top: Top of the directory tree
            exts: List of files extensions. if exts == "all_files"
                    we try to open all files in top
            exclude_dirs: Wildcard used to exclude directories.

        Returns:
            PseudoTable sorted by atomic number Z.
        """
    pseudos = []
    if exts == 'all_files':
        for filepath in [os.path.join(top, fn) for fn in os.listdir(top)]:
            if os.path.isfile(filepath):
                try:
                    pseudo = Pseudo.from_file(filepath)
                    if pseudo:
                        pseudos.append(pseudo)
                    else:
                        logger.info(f'Skipping file {filepath}')
                except Exception:
                    logger.info(f'Skipping file {filepath}')
        if not pseudos:
            logger.warning(f'No pseudopotentials parsed from folder {top}')
            return None
        logger.info(f'Creating PseudoTable with {len(pseudos)} pseudopotentials')
    else:
        if exts is None:
            exts = ('psp8',)
        for pseudo in find_exts(top, exts, exclude_dirs=exclude_dirs):
            try:
                pseudos.append(Pseudo.from_file(pseudo))
            except Exception as exc:
                logger.critical(f'Error in {pseudo}:\n{exc}')
    return cls(pseudos).sort_by_z()