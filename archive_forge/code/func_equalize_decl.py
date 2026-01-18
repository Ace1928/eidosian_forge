from __future__ import annotations
from io import (
import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.io.common import get_handle
from pandas.io.xml import read_xml
def equalize_decl(doc):
    if doc is not None:
        doc = doc.replace('<?xml version="1.0" encoding="utf-8"?', "<?xml version='1.0' encoding='utf-8'?")
    return doc