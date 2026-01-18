from __future__ import division
from builtins import next
from builtins import zip
from builtins import range
import sys
import inspect
import numpy
from numpy.core import numeric
import uncertainties.umath_core as umath_core
import uncertainties.core as uncert_core
from uncertainties.core import deprecation
def func_with_deriv_to_uncert_func(func_with_derivatives):
    """
    Return a function that can be applied to array-like objects that
    contain numbers with uncertainties (lists, lists of lists, NumPy
    arrays, etc.).

    func_with_derivatives -- defines a function that takes an
    array-like object containing scalars and returns an array.  Both
    the value and the derivatives of this function with respect to
    multiple scalar parameters are calculated by this
    func_with_derivatives() argument.

    func_with_derivatives(arr, input_type, derivatives, *args,
    **kwargs) must return an iterator.  The first element returned by
    this iterator is the value of the function at the n-dimensional
    array-like 'arr' (with the correct type).  The following elements
    are arrays that represent the derivative of the function for each
    derivative array from the iterator 'derivatives'.

    func_with_derivatives() takes the following arguments:

      arr -- NumPy ndarray of scalars where the function must be
      evaluated.

      input_type -- data type of the input array-like object.  This
      type is used for determining the type that the function should
      return.

      derivatives -- iterator that returns the derivatives of the
      argument of the function with respect to multiple scalar
      variables.  func_with_derivatives() returns the derivatives of
      the defined function with respect to these variables.

      args -- additional arguments that define the result (example:
      for the pseudo-inverse numpy.linalg.pinv: numerical cutoff).

    Examples of func_with_derivatives: inv_with_derivatives().
    """

    def wrapped_func(array_like, *args, **kwargs):
        """
        array_like -- n-dimensional array-like object that contains
        numbers with uncertainties (list, NumPy ndarray or matrix,
        etc.).

        args -- additional arguments that are passed directly to
        func_with_derivatives.
        """
        array_version = numpy.asanyarray(array_like)
        variables = set()
        for element in array_version.flat:
            if isinstance(element, uncert_core.AffineScalarFunc):
                variables |= set(element.derivatives.keys())
        array_nominal = nominal_values(array_version)
        func_then_derivs = func_with_derivatives(array_nominal, type(array_like), (array_derivative(array_version, var) for var in variables), *args, **kwargs)
        func_nominal_value = next(func_then_derivs)
        if not variables:
            return func_nominal_value
        derivatives = numpy.array([{} for _ in range(func_nominal_value.size)], dtype=object).reshape(func_nominal_value.shape)
        for var, deriv_wrt_var in zip(variables, func_then_derivs):
            for derivative_dict, derivative_value in zip(derivatives.flat, deriv_wrt_var.flat):
                if derivative_value:
                    derivative_dict[var] = derivative_value
        result = numpy.vectorize(uncert_core.AffineScalarFunc)(func_nominal_value, numpy.vectorize(uncert_core.LinearCombination)(derivatives))
        if isinstance(result, numpy.matrix):
            result = result.view(matrix)
        return result
    return wrapped_func