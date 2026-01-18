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
def simple_name_generator(obj):
    """
    Simple name_generator designed for HoloViews objects.

    Objects are labeled with {group}-{label} for each nested
    object, based on a depth-first search.  Adjacent objects with
    identical representations yield only a single copy of the
    representation, to avoid long names for the common case of
    a container whose element(s) share the same group and label.
    """
    if isinstance(obj, LabelledData):
        labels = obj.traverse(lambda x: x.group + ('-' + x.label if x.label else ''))
        labels = [l[0] for l in itertools.groupby(labels)]
        obj_str = ','.join(labels)
    else:
        obj_str = repr(obj)
    return obj_str