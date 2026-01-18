import os
import threading
from contextlib import contextmanager as contextmanager
@contextmanager
def config_context(*, assume_finite=None, working_memory=None, print_changed_only=None, display=None, pairwise_dist_chunk_size=None, enable_cython_pairwise_dist=None, array_api_dispatch=None, transform_output=None, enable_metadata_routing=None, skip_parameter_validation=None):
    """Context manager for global scikit-learn configuration.

    Parameters
    ----------
    assume_finite : bool, default=None
        If True, validation for finiteness will be skipped,
        saving time, but leading to potential crashes. If
        False, validation for finiteness will be performed,
        avoiding error. If None, the existing value won't change.
        The default value is False.

    working_memory : int, default=None
        If set, scikit-learn will attempt to limit the size of temporary arrays
        to this number of MiB (per job when parallelised), often saving both
        computation time and memory on expensive operations that can be
        performed in chunks. If None, the existing value won't change.
        The default value is 1024.

    print_changed_only : bool, default=None
        If True, only the parameters that were set to non-default
        values will be printed when printing an estimator. For example,
        ``print(SVC())`` while True will only print 'SVC()', but would print
        'SVC(C=1.0, cache_size=200, ...)' with all the non-changed parameters
        when False. If None, the existing value won't change.
        The default value is True.

        .. versionchanged:: 0.23
           Default changed from False to True.

    display : {'text', 'diagram'}, default=None
        If 'diagram', estimators will be displayed as a diagram in a Jupyter
        lab or notebook context. If 'text', estimators will be displayed as
        text. If None, the existing value won't change.
        The default value is 'diagram'.

        .. versionadded:: 0.23

    pairwise_dist_chunk_size : int, default=None
        The number of row vectors per chunk for the accelerated pairwise-
        distances reduction backend. Default is 256 (suitable for most of
        modern laptops' caches and architectures).

        Intended for easier benchmarking and testing of scikit-learn internals.
        End users are not expected to benefit from customizing this configuration
        setting.

        .. versionadded:: 1.1

    enable_cython_pairwise_dist : bool, default=None
        Use the accelerated pairwise-distances reduction backend when
        possible. Global default: True.

        Intended for easier benchmarking and testing of scikit-learn internals.
        End users are not expected to benefit from customizing this configuration
        setting.

        .. versionadded:: 1.1

    array_api_dispatch : bool, default=None
        Use Array API dispatching when inputs follow the Array API standard.
        Default is False.

        See the :ref:`User Guide <array_api>` for more details.

        .. versionadded:: 1.2

    transform_output : str, default=None
        Configure output of `transform` and `fit_transform`.

        See :ref:`sphx_glr_auto_examples_miscellaneous_plot_set_output.py`
        for an example on how to use the API.

        - `"default"`: Default output format of a transformer
        - `"pandas"`: DataFrame output
        - `"polars"`: Polars output
        - `None`: Transform configuration is unchanged

        .. versionadded:: 1.2
        .. versionadded:: 1.4
            `"polars"` option was added.

    enable_metadata_routing : bool, default=None
        Enable metadata routing. By default this feature is disabled.

        Refer to :ref:`metadata routing user guide <metadata_routing>` for more
        details.

        - `True`: Metadata routing is enabled
        - `False`: Metadata routing is disabled, use the old syntax.
        - `None`: Configuration is unchanged

        .. versionadded:: 1.3

    skip_parameter_validation : bool, default=None
        If `True`, disable the validation of the hyper-parameters' types and values in
        the fit method of estimators and for arguments passed to public helper
        functions. It can save time in some situations but can lead to low level
        crashes and exceptions with confusing error messages.

        Note that for data parameters, such as `X` and `y`, only type validation is
        skipped but validation with `check_array` will continue to run.

        .. versionadded:: 1.3

    Yields
    ------
    None.

    See Also
    --------
    set_config : Set global scikit-learn configuration.
    get_config : Retrieve current values of the global configuration.

    Notes
    -----
    All settings, not just those presently modified, will be returned to
    their previous values when the context manager is exited.

    Examples
    --------
    >>> import sklearn
    >>> from sklearn.utils.validation import assert_all_finite
    >>> with sklearn.config_context(assume_finite=True):
    ...     assert_all_finite([float('nan')])
    >>> with sklearn.config_context(assume_finite=True):
    ...     with sklearn.config_context(assume_finite=False):
    ...         assert_all_finite([float('nan')])
    Traceback (most recent call last):
    ...
    ValueError: Input contains NaN...
    """
    old_config = get_config()
    set_config(assume_finite=assume_finite, working_memory=working_memory, print_changed_only=print_changed_only, display=display, pairwise_dist_chunk_size=pairwise_dist_chunk_size, enable_cython_pairwise_dist=enable_cython_pairwise_dist, array_api_dispatch=array_api_dispatch, transform_output=transform_output, enable_metadata_routing=enable_metadata_routing, skip_parameter_validation=skip_parameter_validation)
    try:
        yield
    finally:
        set_config(**old_config)