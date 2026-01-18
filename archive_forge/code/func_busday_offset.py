import functools
from . import overrides
from . import _multiarray_umath
from ._multiarray_umath import *  # noqa: F403
from ._multiarray_umath import (
from_dlpack.__module__ = 'numpy'
frombuffer.__module__ = 'numpy'
fromfile.__module__ = 'numpy'
fromiter.__module__ = 'numpy'
frompyfunc.__module__ = 'numpy'
fromstring.__module__ = 'numpy'
@array_function_from_c_func_and_dispatcher(_multiarray_umath.busday_offset)
def busday_offset(dates, offsets, roll=None, weekmask=None, holidays=None, busdaycal=None, out=None):
    """
    busday_offset(dates, offsets, roll='raise', weekmask='1111100', holidays=None, busdaycal=None, out=None)

    First adjusts the date to fall on a valid day according to
    the ``roll`` rule, then applies offsets to the given dates
    counted in valid days.

    .. versionadded:: 1.7.0

    Parameters
    ----------
    dates : array_like of datetime64[D]
        The array of dates to process.
    offsets : array_like of int
        The array of offsets, which is broadcast with ``dates``.
    roll : {'raise', 'nat', 'forward', 'following', 'backward', 'preceding', 'modifiedfollowing', 'modifiedpreceding'}, optional
        How to treat dates that do not fall on a valid day. The default
        is 'raise'.

          * 'raise' means to raise an exception for an invalid day.
          * 'nat' means to return a NaT (not-a-time) for an invalid day.
          * 'forward' and 'following' mean to take the first valid day
            later in time.
          * 'backward' and 'preceding' mean to take the first valid day
            earlier in time.
          * 'modifiedfollowing' means to take the first valid day
            later in time unless it is across a Month boundary, in which
            case to take the first valid day earlier in time.
          * 'modifiedpreceding' means to take the first valid day
            earlier in time unless it is across a Month boundary, in which
            case to take the first valid day later in time.
    weekmask : str or array_like of bool, optional
        A seven-element array indicating which of Monday through Sunday are
        valid days. May be specified as a length-seven list or array, like
        [1,1,1,1,1,0,0]; a length-seven string, like '1111100'; or a string
        like "Mon Tue Wed Thu Fri", made up of 3-character abbreviations for
        weekdays, optionally separated by white space. Valid abbreviations
        are: Mon Tue Wed Thu Fri Sat Sun
    holidays : array_like of datetime64[D], optional
        An array of dates to consider as invalid dates.  They may be
        specified in any order, and NaT (not-a-time) dates are ignored.
        This list is saved in a normalized form that is suited for
        fast calculations of valid days.
    busdaycal : busdaycalendar, optional
        A `busdaycalendar` object which specifies the valid days. If this
        parameter is provided, neither weekmask nor holidays may be
        provided.
    out : array of datetime64[D], optional
        If provided, this array is filled with the result.

    Returns
    -------
    out : array of datetime64[D]
        An array with a shape from broadcasting ``dates`` and ``offsets``
        together, containing the dates with offsets applied.

    See Also
    --------
    busdaycalendar : An object that specifies a custom set of valid days.
    is_busday : Returns a boolean array indicating valid days.
    busday_count : Counts how many valid days are in a half-open date range.

    Examples
    --------
    >>> # First business day in October 2011 (not accounting for holidays)
    ... np.busday_offset('2011-10', 0, roll='forward')
    numpy.datetime64('2011-10-03')
    >>> # Last business day in February 2012 (not accounting for holidays)
    ... np.busday_offset('2012-03', -1, roll='forward')
    numpy.datetime64('2012-02-29')
    >>> # Third Wednesday in January 2011
    ... np.busday_offset('2011-01', 2, roll='forward', weekmask='Wed')
    numpy.datetime64('2011-01-19')
    >>> # 2012 Mother's Day in Canada and the U.S.
    ... np.busday_offset('2012-05', 1, roll='forward', weekmask='Sun')
    numpy.datetime64('2012-05-13')

    >>> # First business day on or after a date
    ... np.busday_offset('2011-03-20', 0, roll='forward')
    numpy.datetime64('2011-03-21')
    >>> np.busday_offset('2011-03-22', 0, roll='forward')
    numpy.datetime64('2011-03-22')
    >>> # First business day after a date
    ... np.busday_offset('2011-03-20', 1, roll='backward')
    numpy.datetime64('2011-03-21')
    >>> np.busday_offset('2011-03-22', 1, roll='backward')
    numpy.datetime64('2011-03-23')
    """
    return (dates, offsets, weekmask, holidays, out)