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
def test_dataclass_json_dict() -> None:

    @dataclasses.dataclass(frozen=True)
    class MyDC:
        q: cirq.LineQubit
        desc: str

        def _json_dict_(self):
            return cirq.dataclass_json_dict(self)

    def custom_resolver(name):
        if name == 'MyDC':
            return MyDC
    my_dc = MyDC(cirq.LineQubit(4), 'hi mom')
    assert_json_roundtrip_works(my_dc, resolvers=[custom_resolver, *cirq.DEFAULT_RESOLVERS])