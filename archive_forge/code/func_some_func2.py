import numpy as np
import pyqtgraph as pg
def some_func2():
    try:
        raise Exception()
    except:
        import sys
        return sys.exc_info()[2]