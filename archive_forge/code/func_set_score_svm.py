import numpy as np
from onnx.reference.ops.aionnxml._common_classifier import (
from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl
from onnx.reference.ops.aionnxml.op_svm_helper import SVMCommon
def set_score_svm(max_weight, maxclass, has_proba, weights_are_all_positive_, classlabels, posclass, negclass):
    write_additional_scores = -1
    if len(classlabels) == 2:
        write_additional_scores = 2
        if not has_proba:
            if weights_are_all_positive_ and max_weight >= 0.5:
                return (classlabels[1], write_additional_scores)
            if max_weight > 0 and (not weights_are_all_positive_):
                return (classlabels[maxclass], write_additional_scores)
        return (classlabels[maxclass], write_additional_scores)
    if max_weight > 0:
        return (posclass, write_additional_scores)
    return (negclass, write_additional_scores)