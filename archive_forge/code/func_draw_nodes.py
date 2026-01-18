import sys
import random
import datetime
import simplejson as json
from collections import OrderedDict
def draw_nodes(start, nodes_list, cores, minute_scale, space_between_minutes, colors):
    """
    Function to return the html-string of the node drawings for the
    gantt chart

    Parameters
    ----------
    start : datetime.datetime obj
        start time for first node
    nodes_list : list
        a list of the node dictionaries
    cores : integer
        the number of cores given to the workflow via the 'n_procs'
        plugin arg
    total_duration : float
        total duration of the workflow execution (in seconds)
    minute_scale : integer
        the scale, in minutes, at which to plot line markers for the
        gantt chart; for example, minute_scale=10 means there are lines
        drawn at every 10 minute interval from start to finish
    space_between_minutes : integer
        scale factor in pixel spacing between minute line markers
    colors : list
        a list of colors to choose from when coloring the nodes in the
        gantt chart

    Returns
    -------
    result : string
        the html-formatted string for producing the minutes-based
        time line markers
    """
    result = ''
    scale = space_between_minutes / minute_scale
    space_between_minutes = space_between_minutes / scale
    end_times = [datetime.datetime(start.year, start.month, start.day, start.hour, start.minute, start.second) for core in range(cores)]
    for node in nodes_list:
        node_start = node['start']
        node_finish = node['finish']
        offset = (node_start - start).total_seconds() / 60 * scale * space_between_minutes + 220
        scale_duration = node['duration'] / 60 * scale * space_between_minutes
        if scale_duration < 5:
            scale_duration = 5
        scale_duration -= 2
        left = 60
        for core in range(len(end_times)):
            if end_times[core] < node_start:
                left += core * 30
                end_times[core] = datetime.datetime(node_finish.year, node_finish.month, node_finish.day, node_finish.hour, node_finish.minute, node_finish.second)
                break
        color = random.choice(colors)
        if 'error' in node:
            color = 'red'
        node_dict = {'left': left, 'offset': offset, 'scale_duration': scale_duration, 'color': color, 'node_name': node['name'], 'node_dur': node['duration'] / 60.0, 'node_start': node_start.strftime('%Y-%m-%d %H:%M:%S'), 'node_finish': node_finish.strftime('%Y-%m-%d %H:%M:%S')}
        new_node = "<div class='node' style='left:%(left)spx;top:%(offset)spx;height:%(scale_duration)spx;background-color:%(color)s;'title='%(node_name)s\nduration:%(node_dur)s\nstart:%(node_start)s\nend:%(node_finish)s'></div>" % node_dict
        result += new_node
    return result