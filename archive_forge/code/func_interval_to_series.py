from collections.abc import MutableMapping
from pyomo.contrib.mpc.data.dynamic_data_base import _is_iterable, _DynamicDataBase
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.interval_data import IntervalData
from pyomo.contrib.mpc.data.find_nearest_index import find_nearest_interval_index
def interval_to_series(data, time_points=None, tolerance=0.0, use_left_endpoints=False, prefer_left=True):
    """
    Arguments
    ---------
    data: IntervalData
        Data to convert to a TimeSeriesData object
    time_points: Iterable (optional)
        Points at which time series will be defined. Values are taken
        from the interval in which each point lives. The default is to
        use the right endpoint of each interval.
    tolerance: Float (optional)
        Tolerance within which time points are considered equal.
        Default is zero.
    use_left_endpoints: Bool (optional)
        Whether the left endpoints should be used in the case when
        time_points is not provided. Default is False, meaning that
        the right interval endpoints will be used. Should not be set
        if time points are provided.
    prefer_left: Bool (optional)
        If time_points is provided, and a time point is equal (within
        tolerance) to a boundary between two intervals, this flag
        controls which interval is used.

    Returns
    -------
    TimeSeriesData

    """
    if time_points is None:
        if use_left_endpoints:
            time_points = [t for t, _ in data.get_intervals()]
        else:
            time_points = [t for _, t in data.get_intervals()]
        series_data = data.get_data()
        return TimeSeriesData(series_data, time_points)
    if use_left_endpoints:
        raise RuntimeError('Cannot provide time_points with use_left_endpoints=True')
    intervals = data.get_intervals()
    data_dict = data.get_data()
    idx_list = [find_nearest_interval_index(intervals, t, tolerance=tolerance, prefer_left=prefer_left) for t in time_points]
    for i, t in enumerate(time_points):
        if idx_list[i] is None:
            raise RuntimeError('Time point %s cannot be found in intervals within tolerance %s.' % (t, tolerance))
    new_data = {key: [vals[i] for i in idx_list] for key, vals in data_dict.items()}
    return TimeSeriesData(new_data, time_points)