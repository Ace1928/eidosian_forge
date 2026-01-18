import numpy.testing
import cupy
def assert_array_equal(x, y, err_msg='', verbose=True, strides_check=False):
    """Raises an AssertionError if two array_like objects are not equal.

    Args:
         x(numpy.ndarray or cupy.ndarray): The actual object to check.
         y(numpy.ndarray or cupy.ndarray): The desired, expected object.
         strides_check(bool): If ``True``, consistency of strides is also
             checked.
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values
             are appended to the error message.

    .. seealso:: :func:`numpy.testing.assert_array_equal`
    """
    numpy.testing.assert_array_equal(cupy.asnumpy(x), cupy.asnumpy(y), err_msg=err_msg, verbose=verbose)
    if strides_check:
        if x.strides != y.strides:
            msg = ['Strides are not equal:']
            if err_msg:
                msg = [msg[0] + ' ' + err_msg]
            if verbose:
                msg.append(' x: {}'.format(x.strides))
                msg.append(' y: {}'.format(y.strides))
            raise AssertionError('\n'.join(msg))