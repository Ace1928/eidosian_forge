import math
def compute_cost(z, print_missed=True):
    C = 0
    missed = {}
    for node in z.descend():
        f = node.fn_name
        if f in COSTS:
            C += COSTS[f](node)
        else:
            missed[f] = missed.get(f, 0) + 1
    if missed and print_missed:
        import warnings
        warnings.warn(f'Missed {missed} in cost computation.')
    return C