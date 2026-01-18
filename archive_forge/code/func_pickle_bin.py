import zlib
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set, Union
import numpy
import srsly
from numpy import ndarray
from thinc.api import NumpyOps
from ..attrs import IDS, ORTH, SPACY, intify_attr
from ..compat import copy_reg
from ..errors import Errors
from ..util import SimpleFrozenList, ensure_path
from ..vocab import Vocab
from ._dict_proxies import SpanGroups
from .doc import DOCBIN_ALL_ATTRS as ALL_ATTRS
from .doc import Doc
def pickle_bin(doc_bin):
    return (unpickle_bin, (doc_bin.to_bytes(),))