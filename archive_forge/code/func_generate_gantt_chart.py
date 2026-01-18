import sys
import random
import datetime
import simplejson as json
from collections import OrderedDict
def generate_gantt_chart(logfile, cores, minute_scale=10, space_between_minutes=50, colors=['#7070FF', '#4E4EB2', '#2D2D66', '#9B9BFF']):
    """
    Generates a gantt chart in html showing the workflow execution based on a callback log file.
    This script was intended to be used with the MultiprocPlugin.
    The following code shows how to set up the workflow in order to generate the log file:

    Parameters
    ----------
    logfile : string
        filepath to the callback log file to plot the gantt chart of
    cores : integer
        the number of cores given to the workflow via the 'n_procs'
        plugin arg
    minute_scale : integer (optional); default=10
        the scale, in minutes, at which to plot line markers for the
        gantt chart; for example, minute_scale=10 means there are lines
        drawn at every 10 minute interval from start to finish
    space_between_minutes : integer (optional); default=50
        scale factor in pixel spacing between minute line markers
    colors : list (optional)
        a list of colors to choose from when coloring the nodes in the
        gantt chart


    Returns
    -------
    None
        the function does not return any value but writes out an html
        file in the same directory as the callback log path passed in

    Usage
    -----
    # import logging
    # import logging.handlers
    # from nipype.utils.profiler import log_nodes_cb

    # log_filename = 'callback.log'
    # logger = logging.getLogger('callback')
    # logger.setLevel(logging.DEBUG)
    # handler = logging.FileHandler(log_filename)
    # logger.addHandler(handler)

    # #create workflow
    # workflow = ...

    # workflow.run(plugin='MultiProc',
    #     plugin_args={'n_procs':8, 'memory':12, 'status_callback': log_nodes_cb})

    # generate_gantt_chart('callback.log', 8)
    """
    html_string = '<!DOCTYPE html>\n    <head>\n        <style>\n            #content{\n                width:99%;\n                height:100%;\n                position:absolute;\n            }\n\n            .node{\n                background-color:#7070FF;\n                border-radius: 5px;\n                position:absolute;\n                width:20px;\n                white-space:pre-wrap;\n            }\n\n            .line{\n                position: absolute;\n                color: #C2C2C2;\n                opacity: 0.5;\n                margin: 0px;\n            }\n\n            .time{\n                position: absolute;\n                font-size: 16px;\n                color: #666666;\n                margin: 0px;\n            }\n\n            .bar{\n                position: absolute;\n                height: 1px;\n                opacity: 0.7;\n            }\n\n            .dot{\n                position: absolute;\n                width: 1px;\n                height: 1px;\n                background-color: red;\n            }\n            .label {\n                width:20px;\n                height:20px;\n                opacity: 0.7;\n                display: inline-block;\n            }\n        </style>\n    </head>\n\n    <body>\n        <div id="content">\n            <div style="display:inline-block;">\n    '
    close_header = '\n    </div>\n    <div style="display:inline-block;margin-left:60px;vertical-align: top;">\n        <p><span><div class="label" style="background-color:#90BBD7;"></div> Estimated Resource</span></p>\n        <p><span><div class="label" style="background-color:#03969D;"></div> Actual Resource</span></p>\n        <p><span><div class="label" style="background-color:#f00;"></div> Failed Node</span></p>\n    </div>\n    '
    nodes_list = log_to_dict(logfile)
    start_node = nodes_list[0]
    last_node = nodes_list[-1]
    duration = (last_node['finish'] - start_node['start']).total_seconds()
    events = create_event_dict(start_node['start'], nodes_list)
    html_string += '<p>Start: ' + start_node['start'].strftime('%Y-%m-%d %H:%M:%S') + '</p>'
    html_string += '<p>Finish: ' + last_node['finish'].strftime('%Y-%m-%d %H:%M:%S') + '</p>'
    html_string += '<p>Duration: ' + '{0:.2f}'.format(duration / 60) + ' minutes</p>'
    html_string += '<p>Nodes: ' + str(len(nodes_list)) + '</p>'
    html_string += '<p>Cores: ' + str(cores) + '</p>'
    html_string += close_header
    html_string += draw_lines(start_node['start'], duration, minute_scale, space_between_minutes)
    html_string += draw_nodes(start_node['start'], nodes_list, cores, minute_scale, space_between_minutes, colors)
    estimated_mem_ts = calculate_resource_timeseries(events, 'estimated_memory_gb')
    runtime_mem_ts = calculate_resource_timeseries(events, 'runtime_memory_gb')
    resource_offset = 120 + 30 * cores
    html_string += draw_resource_bar(start_node['start'], last_node['finish'], estimated_mem_ts, space_between_minutes, minute_scale, '#90BBD7', resource_offset * 2 + 120, 'Memory')
    html_string += draw_resource_bar(start_node['start'], last_node['finish'], runtime_mem_ts, space_between_minutes, minute_scale, '#03969D', resource_offset * 2 + 120, 'Memory')
    estimated_threads_ts = calculate_resource_timeseries(events, 'estimated_threads')
    runtime_threads_ts = calculate_resource_timeseries(events, 'runtime_threads')
    html_string += draw_resource_bar(start_node['start'], last_node['finish'], estimated_threads_ts, space_between_minutes, minute_scale, '#90BBD7', resource_offset, 'Threads')
    html_string += draw_resource_bar(start_node['start'], last_node['finish'], runtime_threads_ts, space_between_minutes, minute_scale, '#03969D', resource_offset, 'Threads')
    html_string += '\n        </div>\n    </body>'
    with open(logfile + '.html', 'w') as html_file:
        html_file.write(html_string)