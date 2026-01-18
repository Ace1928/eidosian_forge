import itertools
import os
import pickle
import re
import shutil
import string
import tarfile
import time
import zipfile
from collections import defaultdict
from hashlib import sha256
from io import BytesIO
import param
from param.parameterized import bothmethod
from .dimension import LabelledData
from .element import Collator, Element
from .ndmapping import NdMapping, UniformNdMapping
from .options import Store
from .overlay import Layout, Overlay
from .util import group_sanitizer, label_sanitizer, unique_iterator
@classmethod
def parse_fields(cls, formatter):
    """Returns the format fields otherwise raise exception"""
    if formatter is None:
        return []
    try:
        parse = list(string.Formatter().parse(formatter))
        return {f for f in list(zip(*parse))[1] if f is not None}
    except Exception as e:
        raise SyntaxError(f'Could not parse formatter {formatter!r}') from e