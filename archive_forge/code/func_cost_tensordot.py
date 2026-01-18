import math
def cost_tensordot(x):
    x1, x2, axes = x.args
    shape1, shape2 = (x1.shape, x2.shape)
    cost = math.prod(shape1) * math.prod(shape2)
    for d in axes[0]:
        cost //= shape1[d]
    return cost