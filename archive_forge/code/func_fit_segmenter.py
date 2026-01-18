from skimage.feature import multiscale_basic_features
def fit_segmenter(labels, features, clf):
    """Segmentation using labeled parts of the image and a classifier.

    Parameters
    ----------
    labels : ndarray of ints
        Image of labels. Labels >= 1 correspond to the training set and
        label 0 to unlabeled pixels to be segmented.
    features : ndarray
        Array of features, with the first dimension corresponding to the number
        of features, and the other dimensions correspond to ``labels.shape``.
    clf : classifier object
        classifier object, exposing a ``fit`` and a ``predict`` method as in
        scikit-learn's API, for example an instance of
        ``RandomForestClassifier`` or ``LogisticRegression`` classifier.

    Returns
    -------
    clf : classifier object
        classifier trained on ``labels``

    Raises
    ------
    NotFittedError if ``self.clf`` has not been fitted yet (use ``self.fit``).
    """
    mask = labels > 0
    training_data = features[mask]
    training_labels = labels[mask].ravel()
    clf.fit(training_data, training_labels)
    return clf