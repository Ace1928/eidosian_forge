import sys
from typing import List, Optional, Sequence
import numpy as np
from onnx import ModelProto
from onnx.backend.test.case.test_case import TestCase
from onnx.backend.test.case.utils import import_recursive
def collect_testcases() -> List[TestCase]:
    """Collect model test cases defined in python/numpy code."""
    real_model_testcases = []
    model_tests = [('test_bvlc_alexnet', 'bvlc_alexnet', 0.001, 1e-07), ('test_densenet121', 'densenet121', 0.002, 1e-07), ('test_inception_v1', 'inception_v1', 0.001, 1e-07), ('test_inception_v2', 'inception_v2', 0.001, 1e-07), ('test_resnet50', 'resnet50', 0.001, 1e-07), ('test_shufflenet', 'shufflenet', 0.001, 1e-07), ('test_squeezenet', 'squeezenet', 0.001, 1e-07), ('test_vgg19', 'vgg19', 0.001, 1e-07), ('test_zfnet512', 'zfnet512', 0.001, 1e-07)]
    for test_name, model_name, rtol, atol in model_tests:
        url = BASE_URL % model_name
        real_model_testcases.append(TestCase(name=test_name, model_name=model_name, url=url, model_dir=None, model=None, data_sets=None, kind='real', rtol=rtol, atol=atol))
    import_recursive(sys.modules[__name__])
    return real_model_testcases + _SimpleModelTestCases