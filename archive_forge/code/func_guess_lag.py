import numpy as np
def guess_lag(x, y):
    """Given two axes returns a guess of the lag point.

    The lag point is defined as the x point where the difference in y
    with the next point is higher then the mean differences between
    the points plus one standard deviation. If such point is not found
    or x and y have different lengths the function returns zero.
    """
    if len(x) != len(y):
        return 0
    diffs = []
    indexes = range(len(x))
    for i in indexes:
        if i + 1 not in indexes:
            continue
        diffs.append(y[i + 1] - y[i])
    diffs = np.array(diffs)
    flex = x[-1]
    for i in indexes:
        if i + 1 not in indexes:
            continue
        if y[i + 1] - y[i] > diffs.mean() + diffs.std():
            flex = x[i]
            break
    return flex