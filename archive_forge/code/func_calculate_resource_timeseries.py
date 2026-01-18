import sys
import random
import datetime
import simplejson as json
from collections import OrderedDict
def calculate_resource_timeseries(events, resource):
    """
    Given as event dictionary, calculate the resources used
    as a timeseries

    Parameters
    ----------
    events : dictionary
        a dictionary of event-based node dictionaries of the workflow
        execution statistics
    resource : string
        the resource of interest to return the time-series of;
        e.g. 'runtime_memory_gb', 'estimated_threads', etc

    Returns
    -------
    time_series : pandas Series
        a pandas Series object that contains timestamps as the indices
        and the resource amount as values
    """
    import pandas as pd
    res = OrderedDict()
    all_res = 0.0
    for _, event in sorted(events.items()):
        if event['event'] == 'start':
            if resource in event and event[resource] != 'Unknown':
                all_res += float(event[resource])
            current_time = event['start']
        elif event['event'] == 'finish':
            if resource in event and event[resource] != 'Unknown':
                all_res -= float(event[resource])
            current_time = event['finish']
        res[current_time] = all_res
    time_series = pd.Series(data=list(res.values()), index=list(res.keys()))
    ts_diff = time_series.diff()
    time_series = time_series[ts_diff != 0]
    return time_series