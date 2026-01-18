import psutil
import panel as pn
import pandas as pd
import holoviews as hv
from holoviews import dim, opts
def get_cpu_data():
    cpu_percent = psutil.cpu_percent(percpu=True)
    df = pd.DataFrame(list(enumerate(cpu_percent)), columns=['CPU', 'Utilization'])
    df['time'] = pd.Timestamp.now()
    return df