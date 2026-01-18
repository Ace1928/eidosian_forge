from functools import wraps
def chuang_f2(individual):
    """Binary deceptive function from : Multivariate Multi-Model Approach for
    Globally Multimodal Problems by Chung-Yao Chuang and Wen-Lian Hsu.

    The function takes individual of 40+1 dimensions and has four global optima
    in [1,1,...,0,0], [0,0,...,1,1], [1,1,...,1] and [0,0,...,0].
    """
    total = 0
    if individual[-2] == 0 and individual[-1] == 0:
        for i in range(0, len(individual) - 2, 8):
            total += inv_trap(individual[i:i + 4]) + inv_trap(individual[i + 4:i + 8])
    elif individual[-2] == 0 and individual[-1] == 1:
        for i in range(0, len(individual) - 2, 8):
            total += inv_trap(individual[i:i + 4]) + trap(individual[i + 4:i + 8])
    elif individual[-2] == 1 and individual[-1] == 0:
        for i in range(0, len(individual) - 2, 8):
            total += trap(individual[i:i + 4]) + inv_trap(individual[i + 4:i + 8])
    else:
        for i in range(0, len(individual) - 2, 8):
            total += trap(individual[i:i + 4]) + trap(individual[i + 4:i + 8])
    return (total,)