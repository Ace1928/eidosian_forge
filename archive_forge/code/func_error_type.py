from collections import Counter
from . import EvaluationMixin, evaluation_io
from ..io import load_key
def error_type(det_key, ann_key, strict_fifth=False):
    """
    Compute the evaluation score and error category for a predicted key
    compared to the annotated key.

    Categories and evaluation scores follow the evaluation strategy used
    for MIREX (see http://music-ir.org/mirex/wiki/2017:Audio_Key_Detection).
    There are two evaluation modes for the 'fifth' category: by default,
    a detection falls into the 'fifth' category if it is the fifth of the
    annotation, or the annotation is the fifth of the detection.
    If `strict_fifth` is `True`, only the former case is considered. This is
    the mode used for MIREX.

    Parameters
    ----------
    det_key : int
        Detected key class.
    ann_key : int
        Annotated key class.
    strict_fifth: bool
        Use strict interpretation of the 'fifth' category, as in MIREX.

    Returns
    -------
    score, category : float, str
        Evaluation score and error category.

    """
    ann_root = ann_key % 12
    ann_mode = ann_key // 12
    det_root = det_key % 12
    det_mode = det_key // 12
    major, minor = (0, 1)
    if det_root == ann_root and det_mode == ann_mode:
        return (1.0, 'correct')
    if det_mode == ann_mode and (det_root - ann_root) % 12 == 7:
        return (0.5, 'fifth')
    if not strict_fifth and (det_mode == ann_mode and (det_root - ann_root) % 12 == 5):
        return (0.5, 'fifth')
    if ann_mode == major and det_mode != ann_mode and ((det_root - ann_root) % 12 == 9):
        return (0.3, 'relative')
    if ann_mode == minor and det_mode != ann_mode and ((det_root - ann_root) % 12 == 3):
        return (0.3, 'relative')
    if det_mode != ann_mode and det_root == ann_root:
        return (0.2, 'parallel')
    else:
        return (0.0, 'other')