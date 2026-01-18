from numba import jit
def make_type_change_mutual(jit=lambda x: x):

    @jit
    def foo(x, y):
        if x > 1 and y > 0:
            return x + bar(x - y, y)
        else:
            return y

    @jit
    def bar(x, y):
        if x > 1 and y > 0:
            return x + foo(x - y, y)
        else:
            return y
    return foo