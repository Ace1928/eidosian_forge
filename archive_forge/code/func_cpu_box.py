import psutil
import panel as pn
import pandas as pd
import holoviews as hv
from holoviews import dim, opts
def cpu_box(data):
    return hv.BoxWhisker(data, 'CPU', 'Utilization', label='CPU Usage')