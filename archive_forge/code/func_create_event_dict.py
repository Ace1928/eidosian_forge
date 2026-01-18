import sys
import random
import datetime
import simplejson as json
from collections import OrderedDict
def create_event_dict(start_time, nodes_list):
    """
    Function to generate a dictionary of event (start/finish) nodes
    from the nodes list

    Parameters
    ----------
    start_time : datetime.datetime
        a datetime object of the pipeline start time
    nodes_list : list
        a list of the node dictionaries that were run in the pipeline

    Returns
    -------
    events : dictionary
        a dictionary where the key is the timedelta from the start of
        the pipeline execution to the value node it accompanies
    """
    import copy
    events = {}
    for node in nodes_list:
        estimated_threads = node.get('num_threads', 1)
        estimated_memory_gb = node.get('estimated_memory_gb', 1.0)
        runtime_threads = node.get('runtime_threads', 0)
        runtime_memory_gb = node.get('runtime_memory_gb', 0.0)
        node['estimated_threads'] = estimated_threads
        node['estimated_memory_gb'] = estimated_memory_gb
        node['runtime_threads'] = runtime_threads
        node['runtime_memory_gb'] = runtime_memory_gb
        start_node = node
        finish_node = copy.deepcopy(node)
        start_node['event'] = 'start'
        finish_node['event'] = 'finish'
        start_delta = (node['start'] - start_time).total_seconds()
        finish_delta = (node['finish'] - start_time).total_seconds()
        if events.get(start_delta) or events.get(finish_delta):
            err_msg = 'Event logged twice or events started at exact same time!'
            raise KeyError(err_msg)
        events[start_delta] = start_node
        events[finish_delta] = finish_node
    return events