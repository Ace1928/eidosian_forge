import math
import string
from datetime import datetime, timedelta
from functools import lru_cache
from itertools import cycle
import numpy as np
import pandas as pd
from .utils import find_package_file
def get_pandas_styler():
    """This function returns a Pandas Styler object

    Cf. https://pandas.pydata.org/docs/user_guide/style.html
    """
    x = np.linspace(0, math.pi, 21)
    df = pd.DataFrame({'sin': np.sin(x), 'cos': np.cos(x)}, index=pd.Index(x, name='alpha'))
    s = df.style
    s.background_gradient(axis=None, cmap='YlOrRd')
    s.format('{:.3f}')
    try:
        s.format_index('{:.3f}')
    except AttributeError:
        pass
    s.set_caption('A Pandas Styler object with background colors and tooltips').set_table_styles([{'selector': 'caption', 'props': 'caption-side: bottom; font-size:1.25em;'}])
    ttips = pd.DataFrame({'sin': ['The sinus of {:.6f} is {:.6f}'.format(t, np.sin(t)) for t in x], 'cos': ['The cosinus of {:.6f} is {:.6f}'.format(t, np.cos(t)) for t in x]}, index=df.index)
    try:
        s.set_tooltips(ttips)
    except AttributeError:
        pass
    return s