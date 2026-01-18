import cupy
def convolve1d2o(in1, in2, mode='valid', method='direct'):
    """
    Convolve a 1-dimensional arrays with a 2nd order filter.
    This results in a second order convolution.

    Convolve `in1` and `in2`, with the output size determined by the
    `mode` argument.

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    method : str {'auto', 'direct', 'fft'}, optional
        A string indicating which method to use to calculate the convolution.

        ``direct``
           The convolution is determined directly from sums, the definition of
           convolution.
        ``fft``
           The Fourier Transform is used to perform the convolution by calling
           `fftconvolve`.
        ``auto``
           Automatically chooses direct or Fourier method based on an estimate
           of which is faster (default).

    Returns
    -------
    out : ndarray
        A 1-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.

    See Also
    --------
    convolve
    convolve1d2o
    convolve1d3o

    Examples
    --------
    Convolution of a 2nd order filter on a 1d signal

    >>> import cusignal as cs
    >>> import numpy as np
    >>> d = 50
    >>> a = np.random.uniform(-1,1,(200))
    >>> b = np.random.uniform(-1,1,(d,d))
    >>> c = cs.convolve1d2o(a,b)

    """
    if in1.ndim != 1:
        raise ValueError('in1 should have one dimension')
    if in2.ndim != 2:
        raise ValueError('in2 should have three dimension')
    if mode in ['same', 'full']:
        raise NotImplementedError('Mode == {} not implemented'.format(mode))
    if method == 'direct':
        return _convolve1d2o(in1, in2, mode)
    else:
        raise NotImplementedError('Only Direct method implemented')