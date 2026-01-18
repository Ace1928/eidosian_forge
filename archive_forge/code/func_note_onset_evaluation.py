from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
from . import (evaluation_io, MultiClassEvaluation, SumEvaluation,
from .onsets import onset_evaluation, OnsetEvaluation
from ..io import load_notes
def note_onset_evaluation(detections, annotations, window=WINDOW):
    """
    Determine the true/false positive/negative note onset detections.

    Parameters
    ----------
    detections : numpy array
        Detected notes.
    annotations : numpy array
        Annotated ground truth notes.
    window : float, optional
        Evaluation window [seconds].

    Returns
    -------
    tp : numpy array, shape (num_tp, 2)
        True positive detections.
    fp : numpy array, shape (num_fp, 2)
        False positive detections.
    tn : numpy array, shape (0, 2)
        True negative detections (empty, see notes).
    fn : numpy array, shape (num_fn, 2)
        False negative detections.
    errors : numpy array, shape (num_tp, 2)
        Errors of the true positive detections wrt. the annotations.

    Notes
    -----
    The expected note row format is:

    'note_time' 'MIDI_note' ['duration' ['MIDI_velocity']]

    The returned true negative array is empty, because we are not interested
    in this class, since it is magnitudes bigger than true positives array.

    """
    detections = np.asarray(detections, dtype=np.float)
    annotations = np.asarray(annotations, dtype=np.float)
    if detections.ndim != 2 or annotations.ndim != 2:
        raise ValueError('detections and annotations must be 2D arrays')
    tp = np.zeros((0, 2))
    fp = np.zeros((0, 2))
    tn = np.zeros((0, 2))
    fn = np.zeros((0, 2))
    errors = np.zeros((0, 2))
    if detections.size == 0 and annotations.size == 0:
        return (tp, fp, tn, fn, errors)
    elif annotations.size == 0:
        return (tp, detections, tn, fn, errors)
    elif detections.size == 0:
        return (tp, tp, tn, annotations, errors)
    detections = detections[:, :2]
    annotations = annotations[:, :2]
    notes = np.unique(np.concatenate((detections[:, 1], annotations[:, 1]))).tolist()
    for note in notes:
        det = detections[detections[:, 1] == note]
        ann = annotations[annotations[:, 1] == note]
        tp_, fp_, _, fn_, err_ = onset_evaluation(det[:, 0], ann[:, 0], window)
        tp = np.vstack((tp, det[np.in1d(det[:, 0], tp_)]))
        fp = np.vstack((fp, det[np.in1d(det[:, 0], fp_)]))
        fn = np.vstack((fn, ann[np.in1d(ann[:, 0], fn_)]))
        err_ = np.vstack((np.array(err_), np.repeat(np.asarray([note]), len(err_)))).T
        errors = np.vstack((errors, err_))
    if len(tp) + len(fp) != len(detections):
        raise AssertionError('bad TP / FP calculation')
    if len(tp) + len(fn) != len(annotations):
        raise AssertionError('bad FN calculation')
    if len(tp) != len(errors):
        raise AssertionError('bad errors calculation')
    errors = errors[tp[:, 0].argsort()]
    tp = tp[tp[:, 0].argsort()]
    fp = fp[fp[:, 0].argsort()]
    fn = fn[fn[:, 0].argsort()]
    return (tp, fp, tn, fn, errors)