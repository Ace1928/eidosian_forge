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
def merge_bins(bins):
    merged = None
    for byte_string in bins:
        if byte_string is not None:
            doc_bin = DocBin(store_user_data=True).from_bytes(byte_string)
            if merged is None:
                merged = doc_bin
            else:
                merged.merge(doc_bin)
    if merged is not None:
        return merged.to_bytes()
    else:
        return b''