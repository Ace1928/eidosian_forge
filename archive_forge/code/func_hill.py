import numpy as np
def hill(x):
    if thr:
        return x ** s / (thr ** s + x ** s)
    else:
        return 1.0 * (x != 0)