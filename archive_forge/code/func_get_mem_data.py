import psutil
import panel as pn
import pandas as pd
import holoviews as hv
from holoviews import dim, opts
def get_mem_data():
    vmem = psutil.virtual_memory()
    df = pd.DataFrame(dict(free=vmem.free / vmem.total, used=vmem.used / vmem.total), index=[pd.Timestamp.now()])
    return df * 100