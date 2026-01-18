import numpy as np
from ...metaarray import MetaArray
def modeFilter(data, window=500, step=None, bins=None):
    """Filter based on histogram-based mode function"""
    d1 = data.view(np.ndarray)
    vals = []
    l2 = int(window / 2.0)
    if step is None:
        step = l2
    i = 0
    while True:
        if i > len(data) - step:
            break
        vals.append(mode(d1[i:i + window], bins))
        i += step
    chunks = [np.linspace(vals[0], vals[0], l2)]
    for i in range(len(vals) - 1):
        chunks.append(np.linspace(vals[i], vals[i + 1], step))
    remain = len(data) - step * (len(vals) - 1) - l2
    chunks.append(np.linspace(vals[-1], vals[-1], remain))
    d2 = np.hstack(chunks)
    if hasattr(data, 'implements') and data.implements('MetaArray'):
        return MetaArray(d2, info=data.infoCopy())
    return d2