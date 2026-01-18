from typing import Callable, List, Tuple, Union
import numpy as np
def pwc(timespan):
    """Takes a time span and returns a callable for creating a function that is piece-wise constant in time. The returned
    function takes arguments ``(p, t)``, where ``p`` is an array that defines the bin values for the function.

    Creates a callable for defining a :class:`~.ParametrizedHamiltonian`.

    Args:
        timespan(Union[float, tuple(float, float)]): The time span defining the region where the function is non-zero.
            If an integer is provided, the time span is defined as ``(0, timespan)``.

    Returns:
        callable: a function that takes two arguments: an ``array`` of trainable parameters, and a ``float`` defining the
        time at which the function is evaluated.

    The convenience function ``pwc`` essentially implements

    .. code-block:: python3

        def pwc(timespan):
            def wrapped(p, t):
                return p[int(t/len(p))]
            return wrapped

    This function can be used to create a parametrized coefficient function that is piece-wise constant
    within the interval ``t``, and 0 outside it.

    When creating the callable, only the time span is passed. The number
    of bins and values for the parameters are set when ``params`` is passed to the callable. Each bin value is set by
    an element of the ``params`` array. The variable ``t`` is used to select the value of the parameter array
    corresponding to the specified time, based on the assigned binning.

    .. code-block:: python3

        params = jnp.array([1, 2, 3, 4, 5])
        time = jnp.linspace(0, 10, 1000)
        timespan=(2, 7)
        y = qml.pulse.pwc(timespan)(params, time)
        plt.plot(time, y, label=f"params={params}, timespan={timespan}")
        plt.legend()
        plt.show()

    .. figure:: ../../_static/pulse/pwc_example.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    .. warning::
        The final time in the time span indicates the time at which the function output switches from params[-1] to 0.
        As such, the above function returns ``5`` for a time slightly smaller than the final time in ``timespan``,
        but it returns ``0`` for the final time itself:

        >>> qml.pulse.pwc(timespan)(params, 6.999999)
        Array(5., dtype=float32)

        >>> qml.pulse.pwc(timespan)(params, 7.)
        Array(0., dtype=float32)

    **Example**

    >>> timespan = (2, 7)
    >>> f1 = qml.pulse.pwc(timespan)
    >>> H = f1 * qml.X(0)

    The resulting function ``f1`` has the call signature ``f1(params, t)``. If passed an array of parameters and
    a time, it will assign the array as the constants in the piece-wise function, and select the constant corresponding
    to the specified time, based on the time interval defined by ``timespan``.

    In the following example, passing an array to ``pwc((2, 7))`` evenly distributes the array values in the
    interval ``t=2`` to ``t=7``. The time ``t`` is then used to select one of the array values based on this distribution.

    >>> H(params=[[11, 12, 13, 14, 15]], t=2.3)
    11.0 * X(0)

    >>> H(params=[[11, 12, 13, 14, 15]], t=2.5) # different time, same bin, same result
    11.0 * X(0)

    >>> H(params=[[11, 12, 13, 14, 15]], t=3.1) # next bin
    12.0 * X(0)

    >>> H(params=[[11, 12, 13, 14, 15]], t=8) # outside the window returns 0
    0.0 * X(0)

    """
    if not has_jax:
        raise ImportError('Module jax is required for any pulse-related convenience function. You can install jax via: pip install jax==0.4.3 jaxlib==0.4.3')
    if isinstance(timespan, (tuple, list)):
        t0, t1 = timespan
    else:
        t0 = 0
        t1 = timespan

    def func(params, t):
        num_bins = len(params)
        params = jnp.concatenate([jnp.array(params), jnp.zeros(1)])
        idx = num_bins / (t1 - t0) * (t - t0)
        idx = jnp.where((idx >= 0) & (idx <= num_bins), jnp.array(idx, dtype=int), -1)
        return params[idx]
    return func