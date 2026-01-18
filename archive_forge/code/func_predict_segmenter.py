from skimage.feature import multiscale_basic_features
def predict_segmenter(features, clf):
    """Segmentation of images using a pretrained classifier.

    Parameters
    ----------
    features : ndarray
        Array of features, with the last dimension corresponding to the number
        of features, and the other dimensions are compatible with the shape of
        the image to segment, or a flattened image.
    clf : classifier object
        trained classifier object, exposing a ``predict`` method as in
        scikit-learn's API, for example an instance of
        ``RandomForestClassifier`` or ``LogisticRegression`` classifier. The
        classifier must be already trained, for example with
        :func:`skimage.future.fit_segmenter`.

    Returns
    -------
    output : ndarray
        Labeled array, built from the prediction of the classifier.
    """
    sh = features.shape
    if features.ndim > 2:
        features = features.reshape((-1, sh[-1]))
    try:
        predicted_labels = clf.predict(features)
    except NotFittedError:
        raise NotFittedError('You must train the classifier `clf` firstfor example with the `fit_segmenter` function.')
    except ValueError as err:
        if err.args and 'x must consist of vectors of length' in err.args[0]:
            raise ValueError(err.args[0] + '\n' + 'Maybe you did not use the same type of features for training the classifier.')
        else:
            raise err
    output = predicted_labels.reshape(sh[:-1])
    return output