import contextlib
import dataclasses
import datetime
import importlib
import io
import json
import os
import pathlib
import sys
import warnings
from typing import Dict, List, Optional, Tuple, Type
from unittest import mock
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import sympy
import cirq
from cirq._compat import proper_eq
from cirq.protocols import json_serialization
from cirq.testing.json import ModuleJsonTestSpec, spec_for, assert_json_roundtrip_works
def test_fail_to_resolve():
    buffer = io.StringIO()
    buffer.write('\n    {\n      "cirq_type": "MyCustomClass",\n      "data": [1, 2, 3]\n    }\n    ')
    buffer.seek(0)
    with pytest.raises(ValueError) as e:
        cirq.read_json(buffer)
    assert e.match("Could not resolve type 'MyCustomClass' during deserialization")