import cupy
def cdist(XA, XB, metric='euclidean', out=None, **kwargs):
    """Compute distance between each pair of the two collections of inputs.

    Args:
        XA (array_like): An :math:`m_A` by :math:`n` array of :math:`m_A`
            original observations in an :math:`n`-dimensional space.
            Inputs are converted to float type.
        XB (array_like): An :math:`m_B` by :math:`n` array of :math:`m_B`
            original observations in an :math:`n`-dimensional space.
            Inputs are converted to float type.
        metric (str, optional): The distance metric to use.
            The distance function can be 'canberra', 'chebyshev',
            'cityblock', 'correlation', 'cosine', 'euclidean', 'hamming',
            'hellinger', 'jensenshannon', 'kl_divergence', 'matching',
            'minkowski', 'russellrao', 'sqeuclidean'.
        out (cupy.ndarray, optional): The output array. If not None, the
            distance matrix Y is stored in this array.
        **kwargs (dict, optional): Extra arguments to `metric`: refer to each
            metric documentation for a list of all possible arguments.
            Some possible arguments:
            p (float): The p-norm to apply for Minkowski, weighted and
            unweighted. Default: 2.0

    Returns:
        Y (cupy.ndarray): A :math:`m_A` by :math:`m_B` distance matrix is
            returned. For each :math:`i` and :math:`j`, the metric
            ``dist(u=XA[i], v=XB[j])`` is computed and stored in the
            :math:`ij` th entry.
    """
    if not pylibraft_available:
        raise RuntimeError('pylibraft is not installed')
    XA = cupy.asarray(XA, dtype='float32')
    XB = cupy.asarray(XB, dtype='float32')
    s = XA.shape
    sB = XB.shape
    if len(s) != 2:
        raise ValueError('XA must be a 2-dimensional array.')
    if len(sB) != 2:
        raise ValueError('XB must be a 2-dimensional array.')
    if s[1] != sB[1]:
        raise ValueError('XA and XB must have the same number of columns (i.e. feature dimension.)')
    mA = s[0]
    mB = sB[0]
    p = kwargs['p'] if 'p' in kwargs else 2.0
    if out is not None:
        if out.dtype != 'float32':
            out = out.astype('float32', copy=False)
        if out.shape != (mA, mB):
            cupy.resize(out, (mA, mB))
        out[:] = 0.0
    if isinstance(metric, str):
        mstr = metric.lower()
        metric_info = _METRIC_ALIAS.get(mstr, None)
        if metric_info is not None:
            output_arr = out if out is not None else cupy.zeros((mA, mB), dtype=XA.dtype)
            pairwise_distance(XA, XB, output_arr, metric, p=p)
            return output_arr
        else:
            raise ValueError('Unknown Distance Metric: %s' % mstr)
    else:
        raise TypeError('2nd argument metric must be a string identifier')