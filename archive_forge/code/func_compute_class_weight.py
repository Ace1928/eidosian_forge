import numpy as np
from scipy import sparse
from ._param_validation import StrOptions, validate_params
@validate_params({'class_weight': [dict, StrOptions({'balanced'}), None], 'classes': [np.ndarray], 'y': ['array-like']}, prefer_skip_nested_validation=True)
def compute_class_weight(class_weight, *, classes, y):
    """Estimate class weights for unbalanced datasets.

    Parameters
    ----------
    class_weight : dict, "balanced" or None
        If "balanced", class weights will be given by
        `n_samples / (n_classes * np.bincount(y))`.
        If a dictionary is given, keys are classes and values are corresponding class
        weights.
        If `None` is given, the class weights will be uniform.

    classes : ndarray
        Array of the classes occurring in the data, as given by
        `np.unique(y_org)` with `y_org` the original class labels.

    y : array-like of shape (n_samples,)
        Array of original class labels per sample.

    Returns
    -------
    class_weight_vect : ndarray of shape (n_classes,)
        Array with `class_weight_vect[i]` the weight for i-th class.

    References
    ----------
    The "balanced" heuristic is inspired by
    Logistic Regression in Rare Events Data, King, Zen, 2001.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.utils.class_weight import compute_class_weight
    >>> y = [1, 1, 1, 1, 0, 0]
    >>> compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
    array([1.5 , 0.75])
    """
    from ..preprocessing import LabelEncoder
    if set(y) - set(classes):
        raise ValueError('classes should include all valid labels that can be in y')
    if class_weight is None or len(class_weight) == 0:
        weight = np.ones(classes.shape[0], dtype=np.float64, order='C')
    elif class_weight == 'balanced':
        le = LabelEncoder()
        y_ind = le.fit_transform(y)
        if not all(np.isin(classes, le.classes_)):
            raise ValueError('classes should have valid labels that are in y')
        recip_freq = len(y) / (len(le.classes_) * np.bincount(y_ind).astype(np.float64))
        weight = recip_freq[le.transform(classes)]
    else:
        weight = np.ones(classes.shape[0], dtype=np.float64, order='C')
        unweighted_classes = []
        for i, c in enumerate(classes):
            if c in class_weight:
                weight[i] = class_weight[c]
            else:
                unweighted_classes.append(c)
        n_weighted_classes = len(classes) - len(unweighted_classes)
        if unweighted_classes and n_weighted_classes != len(class_weight):
            unweighted_classes_user_friendly_str = np.array(unweighted_classes).tolist()
            raise ValueError(f'The classes, {unweighted_classes_user_friendly_str}, are not in class_weight')
    return weight