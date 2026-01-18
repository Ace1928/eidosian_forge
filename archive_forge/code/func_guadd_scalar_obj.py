from numba import vectorize
def guadd_scalar_obj(a, b, c):
    Dummy()
    x, y = c.shape
    for i in range(x):
        for j in range(y):
            c[i, j] = a[i, j] + b