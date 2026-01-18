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
def test_context_serialization():

    def custom_resolver(name):
        if name == 'SBKImpl':
            return SBKImpl
    test_resolvers = [custom_resolver] + cirq.DEFAULT_RESOLVERS
    sbki_empty = SBKImpl('sbki_empty')
    assert_json_roundtrip_works(sbki_empty, resolvers=test_resolvers)
    sbki_list = SBKImpl('sbki_list', data_list=[sbki_empty, sbki_empty])
    assert_json_roundtrip_works(sbki_list, resolvers=test_resolvers)
    sbki_tuple = SBKImpl('sbki_tuple', data_tuple=(sbki_list, sbki_list))
    assert_json_roundtrip_works(sbki_tuple, resolvers=test_resolvers)
    sbki_dict = SBKImpl('sbki_dict', data_dict={'a': sbki_tuple, 'b': sbki_tuple})
    assert_json_roundtrip_works(sbki_dict, resolvers=test_resolvers)
    sbki_json = str(cirq.to_json(sbki_dict))
    assert sbki_json.count('"cirq_type": "_SerializedContext"') == 4
    assert sbki_json.count('"cirq_type": "_SerializedKey"') == 7
    final_obj_idx = sbki_json.rfind('{')
    final_obj = sbki_json[final_obj_idx:sbki_json.find('}', final_obj_idx) + 1]
    assert final_obj == '{\n      "cirq_type": "_SerializedKey",\n      "key": 4\n    }'
    list_sbki = [sbki_dict]
    assert_json_roundtrip_works(list_sbki, resolvers=test_resolvers)
    dict_sbki = {'a': sbki_dict}
    assert_json_roundtrip_works(dict_sbki, resolvers=test_resolvers)
    assert sbki_list != json_serialization._SerializedKey(sbki_list)
    sbki_other_list = SBKImpl('sbki_list', data_list=[sbki_list])
    assert_json_roundtrip_works(sbki_other_list, resolvers=test_resolvers)