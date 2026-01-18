import psutil
import panel as pn
import pandas as pd
import holoviews as hv
from holoviews import dim, opts
def mem_stack(data):
    data = pd.melt(data, 'index', var_name='Type', value_name='Usage')
    areas = hv.Dataset(data).to(hv.Area, 'index', 'Usage')
    return hv.Area.stack(areas.overlay()).relabel('Memory')