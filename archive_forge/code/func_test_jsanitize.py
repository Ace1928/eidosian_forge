from __future__ import annotations
import dataclasses
import datetime
import json
import os
import pathlib
from enum import Enum
import numpy as np
import pandas as pd
import pytest
import torch
from bson.objectid import ObjectId
from monty.json import MontyDecoder, MontyEncoder, MSONable, _load_redirect, jsanitize
from . import __version__ as tests_version
def test_jsanitize(self):
    d = {'hello': 1, 'world': None}
    clean = jsanitize(d)
    assert clean['world'] is None
    assert json.loads(json.dumps(d)) == json.loads(json.dumps(clean))
    d = {'hello': GoodMSONClass(1, 2, 3)}
    with pytest.raises(TypeError):
        json.dumps(d)
    clean = jsanitize(d)
    assert isinstance(clean['hello'], str)
    clean_strict = jsanitize(d, strict=True)
    assert clean_strict['hello']['a'] == 1
    assert clean_strict['hello']['b'] == 2
    clean_recursive_msonable = jsanitize(d, recursive_msonable=True)
    assert clean_recursive_msonable['hello']['a'] == 1
    assert clean_recursive_msonable['hello']['b'] == 2
    d = {'dt': datetime.datetime.now()}
    clean = jsanitize(d)
    assert isinstance(clean['dt'], str)
    clean = jsanitize(d, allow_bson=True)
    assert isinstance(clean['dt'], datetime.datetime)
    d = {'a': ['b', np.array([1, 2, 3])], 'b': ObjectId.from_datetime(datetime.datetime.now())}
    clean = jsanitize(d)
    assert clean['a'], ['b', [1, 2 == 3]]
    assert isinstance(clean['b'], str)
    rnd_bin = bytes(np.random.rand(10))
    d = {'a': bytes(rnd_bin)}
    clean = jsanitize(d, allow_bson=True)
    assert clean['a'] == bytes(rnd_bin)
    assert isinstance(clean['a'], bytes)
    p = pathlib.Path('/home/user/')
    clean = jsanitize(p, strict=True)
    assert clean, ['/home/user' in '\\home\\user']
    instance = MethodSerializationClass(a=1)
    for function in [str, list, sum, open, os.path.join, my_callable, MethodSerializationClass.NestedClass.inner_method, MethodSerializationClass.staticmethod, instance.staticmethod, MethodSerializationClass.classmethod, instance.classmethod, MethodSerializationClass, Enum]:
        d = {'f': function}
        clean = jsanitize(d)
        assert '@module' in clean['f']
        assert '@callable' in clean['f']
    for function in [instance.method]:
        d = {'f': function}
        clean = jsanitize(d)
        assert '@module' in clean['f']
        assert '@callable' in clean['f']
        assert clean['f'].get('@bound', None) is not None
        assert '@class' in clean['f']['@bound']
    for function in [MethodNonSerializationClass(1).method]:
        d = {'f': function}
        clean = jsanitize(d)
        assert isinstance(clean['f'], str)
        with pytest.raises(AttributeError):
            jsanitize(d, strict=True)
    d = {'c': instance}
    clean = jsanitize(d, strict=True)
    assert '@class' in clean['c']
    df = pd.DataFrame([{'a': 1, 'b': 1}, {'a': 1, 'b': 2}])
    clean = jsanitize(df)
    assert clean == df.to_dict()
    s = pd.Series({'a': [1, 2, 3], 'b': [4, 5, 6]})
    clean = jsanitize(s)
    assert clean == s.to_dict()