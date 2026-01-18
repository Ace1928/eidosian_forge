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
def test_basic_time_assertions():
    naive_dt = datetime.datetime.now()
    utc_dt = naive_dt.astimezone(datetime.timezone.utc)
    assert naive_dt.timestamp() == utc_dt.timestamp()
    re_utc = datetime.datetime.fromtimestamp(utc_dt.timestamp())
    re_naive = datetime.datetime.fromtimestamp(naive_dt.timestamp())
    assert re_utc == re_naive, 'roundtripping w/o tz turns to naive utc'
    assert re_utc != utc_dt, 'roundtripping loses tzinfo'
    assert naive_dt == re_naive, 'works, as long as you called fromtimestamp from the same timezone'