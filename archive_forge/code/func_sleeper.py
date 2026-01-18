import time
from IPython.lib import backgroundjobs as bg
def sleeper(interval=t_short, *a, **kw):
    args = dict(interval=interval, other_args=a, kw_args=kw)
    time.sleep(interval)
    return args