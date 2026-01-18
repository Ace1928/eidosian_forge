from __future__ import annotations
import functools
import glob
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
import unittest
from collections import defaultdict
from typing import Any, Callable, Iterable, Pattern, Sequence
from urllib.request import urlretrieve
import numpy as np
import onnx
import onnx.reference
from onnx import ONNX_ML, ModelProto, NodeProto, TypeProto, ValueInfoProto, numpy_helper
from onnx.backend.base import Backend
from onnx.backend.test.case.test_case import TestCase
from onnx.backend.test.loader import load_model_tests
from onnx.backend.test.runner.item import TestItem
@classmethod
def prepare_model_data(cls, model_test: TestCase) -> str:
    onnx_home = os.path.expanduser(os.getenv('ONNX_HOME', os.path.join('~', '.onnx')))
    models_dir = os.getenv('ONNX_MODELS', os.path.join(onnx_home, 'models'))
    model_dir: str = os.path.join(models_dir, model_test.model_name)
    if not os.path.exists(os.path.join(model_dir, 'model.onnx')):
        if os.path.exists(model_dir):
            bi = 0
            while True:
                dest = f'{model_dir}.old.{bi}'
                if os.path.exists(dest):
                    bi += 1
                    continue
                shutil.move(model_dir, dest)
                break
        os.makedirs(model_dir)
        cls.download_model(model_test=model_test, model_dir=model_dir, models_dir=models_dir)
    return model_dir