from numpy.testing import dec
from nibabel.data import DataError
def make_label_dec(label, ds=None):
    """Factory function to create a decorator that applies one or more labels.

    Parameters
    ----------
    label : str or sequence
        One or more labels that will be applied by the decorator to the
        functions it decorates. Labels are attributes of the decorated function
        with their value set to True.
    ds : str
        An optional docstring for the resulting decorator. If not given, a
        default docstring is auto-generated.

    Returns
    -------
    ldec : function
        A decorator.

    Examples
    --------
    >>> slow = make_label_dec('slow')
    >>> slow.__doc__
    "Labels a test as 'slow'"

    >>> rare = make_label_dec(['slow','hard'],
    ... "Mix labels 'slow' and 'hard' for rare tests")
    >>> @rare
    ... def f(): pass
    ...
    >>>
    >>> f.slow
    True
    >>> f.hard
    True
    """
    if isinstance(label, str):
        labels = [label]
    else:
        labels = label
    tmp = lambda: None
    for label in labels:
        setattr(tmp, label, True)

    def decor(f):
        for label in labels:
            setattr(f, label, True)
        return f
    if ds is None:
        ds = 'Labels a test as %r' % label
        decor.__doc__ = ds
    return decor