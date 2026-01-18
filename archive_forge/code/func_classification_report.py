import warnings
from numbers import Integral, Real
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.special import xlogy
from ..exceptions import UndefinedMetricWarning
from ..preprocessing import LabelBinarizer, LabelEncoder
from ..utils import (
from ..utils._array_api import _union1d, _weighted_sum, get_namespace
from ..utils._param_validation import Interval, Options, StrOptions, validate_params
from ..utils.extmath import _nanaverage
from ..utils.multiclass import type_of_target, unique_labels
from ..utils.sparsefuncs import count_nonzero
from ..utils.validation import _check_pos_label_consistency, _num_samples
@validate_params({'y_true': ['array-like', 'sparse matrix'], 'y_pred': ['array-like', 'sparse matrix'], 'labels': ['array-like', None], 'target_names': ['array-like', None], 'sample_weight': ['array-like', None], 'digits': [Interval(Integral, 0, None, closed='left')], 'output_dict': ['boolean'], 'zero_division': [Options(Real, {0.0, 1.0}), 'nan', StrOptions({'warn'})]}, prefer_skip_nested_validation=True)
def classification_report(y_true, y_pred, *, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn'):
    """Build a text report showing the main classification metrics.

    Read more in the :ref:`User Guide <classification_report>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.

    labels : array-like of shape (n_labels,), default=None
        Optional list of label indices to include in the report.

    target_names : array-like of shape (n_labels,), default=None
        Optional display names matching the labels (same order).

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    digits : int, default=2
        Number of digits for formatting output floating point values.
        When ``output_dict`` is ``True``, this will be ignored and the
        returned values will not be rounded.

    output_dict : bool, default=False
        If True, return output as dict.

        .. versionadded:: 0.20

    zero_division : {"warn", 0.0, 1.0, np.nan}, default="warn"
        Sets the value to return when there is a zero division. If set to
        "warn", this acts as 0, but warnings are also raised.

        .. versionadded:: 1.3
           `np.nan` option was added.

    Returns
    -------
    report : str or dict
        Text summary of the precision, recall, F1 score for each class.
        Dictionary returned if output_dict is True. Dictionary has the
        following structure::

            {'label 1': {'precision':0.5,
                         'recall':1.0,
                         'f1-score':0.67,
                         'support':1},
             'label 2': { ... },
              ...
            }

        The reported averages include macro average (averaging the unweighted
        mean per label), weighted average (averaging the support-weighted mean
        per label), and sample average (only for multilabel classification).
        Micro average (averaging the total true positives, false negatives and
        false positives) is only shown for multi-label or multi-class
        with a subset of classes, because it corresponds to accuracy
        otherwise and would be the same for all metrics.
        See also :func:`precision_recall_fscore_support` for more details
        on averages.

        Note that in binary classification, recall of the positive class
        is also known as "sensitivity"; recall of the negative class is
        "specificity".

    See Also
    --------
    precision_recall_fscore_support: Compute precision, recall, F-measure and
        support for each class.
    confusion_matrix: Compute confusion matrix to evaluate the accuracy of a
        classification.
    multilabel_confusion_matrix: Compute a confusion matrix for each class or sample.

    Examples
    --------
    >>> from sklearn.metrics import classification_report
    >>> y_true = [0, 1, 2, 2, 2]
    >>> y_pred = [0, 0, 2, 2, 1]
    >>> target_names = ['class 0', 'class 1', 'class 2']
    >>> print(classification_report(y_true, y_pred, target_names=target_names))
                  precision    recall  f1-score   support
    <BLANKLINE>
         class 0       0.50      1.00      0.67         1
         class 1       0.00      0.00      0.00         1
         class 2       1.00      0.67      0.80         3
    <BLANKLINE>
        accuracy                           0.60         5
       macro avg       0.50      0.56      0.49         5
    weighted avg       0.70      0.60      0.61         5
    <BLANKLINE>
    >>> y_pred = [1, 1, 0]
    >>> y_true = [1, 1, 1]
    >>> print(classification_report(y_true, y_pred, labels=[1, 2, 3]))
                  precision    recall  f1-score   support
    <BLANKLINE>
               1       1.00      0.67      0.80         3
               2       0.00      0.00      0.00         0
               3       0.00      0.00      0.00         0
    <BLANKLINE>
       micro avg       1.00      0.67      0.80         3
       macro avg       0.33      0.22      0.27         3
    weighted avg       1.00      0.67      0.80         3
    <BLANKLINE>
    """
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    if labels is None:
        labels = unique_labels(y_true, y_pred)
        labels_given = False
    else:
        labels = np.asarray(labels)
        labels_given = True
    micro_is_accuracy = (y_type == 'multiclass' or y_type == 'binary') and (not labels_given or set(labels) == set(unique_labels(y_true, y_pred)))
    if target_names is not None and len(labels) != len(target_names):
        if labels_given:
            warnings.warn('labels size, {0}, does not match size of target_names, {1}'.format(len(labels), len(target_names)))
        else:
            raise ValueError('Number of classes, {0}, does not match size of target_names, {1}. Try specifying the labels parameter'.format(len(labels), len(target_names)))
    if target_names is None:
        target_names = ['%s' % l for l in labels]
    headers = ['precision', 'recall', 'f1-score', 'support']
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, sample_weight=sample_weight, zero_division=zero_division)
    rows = zip(target_names, p, r, f1, s)
    if y_type.startswith('multilabel'):
        average_options = ('micro', 'macro', 'weighted', 'samples')
    else:
        average_options = ('micro', 'macro', 'weighted')
    if output_dict:
        report_dict = {label[0]: label[1:] for label in rows}
        for label, scores in report_dict.items():
            report_dict[label] = dict(zip(headers, [float(i) for i in scores]))
    else:
        longest_last_line_heading = 'weighted avg'
        name_width = max((len(cn) for cn in target_names))
        width = max(name_width, len(longest_last_line_heading), digits)
        head_fmt = '{:>{width}s} ' + ' {:>9}' * len(headers)
        report = head_fmt.format('', *headers, width=width)
        report += '\n\n'
        row_fmt = '{:>{width}s} ' + ' {:>9.{digits}f}' * 3 + ' {:>9}\n'
        for row in rows:
            report += row_fmt.format(*row, width=width, digits=digits)
        report += '\n'
    for average in average_options:
        if average.startswith('micro') and micro_is_accuracy:
            line_heading = 'accuracy'
        else:
            line_heading = average + ' avg'
        avg_p, avg_r, avg_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=average, sample_weight=sample_weight, zero_division=zero_division)
        avg = [avg_p, avg_r, avg_f1, np.sum(s)]
        if output_dict:
            report_dict[line_heading] = dict(zip(headers, [float(i) for i in avg]))
        elif line_heading == 'accuracy':
            row_fmt_accuracy = '{:>{width}s} ' + ' {:>9.{digits}}' * 2 + ' {:>9.{digits}f}' + ' {:>9}\n'
            report += row_fmt_accuracy.format(line_heading, '', '', *avg[2:], width=width, digits=digits)
        else:
            report += row_fmt.format(line_heading, *avg, width=width, digits=digits)
    if output_dict:
        if 'accuracy' in report_dict.keys():
            report_dict['accuracy'] = report_dict['accuracy']['precision']
        return report_dict
    else:
        return report