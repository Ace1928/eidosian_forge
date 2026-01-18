import numpy as np
from onnx.reference.ops.aionnxml._common_classifier import (
from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl
from onnx.reference.ops.aionnxml.op_svm_helper import SVMCommon
def sigmoid_probability(score, proba, probb):
    val = score * proba + probb
    return 1 - compute_logistic(val)