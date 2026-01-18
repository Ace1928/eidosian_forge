import math
def compute_cost_scalings(z, factor_map, print_missed=True):
    counts = {}
    missed = {}
    for node in z.descend():
        f = node.fn_name
        if f in COST_SCALINGS:
            CS = COST_SCALINGS[f](node)
        else:
            missed[f] = missed.get(f, 0) + 1
            continue
        key = (CS, f)
        counts[key] = counts.get(key, 0) + 1
    if missed and print_missed:
        import warnings
        warnings.warn(f'Missed {missed} in cost scaling computation.')
    scalings = []
    for key, freq in counts.items():
        op = {'cost': key[0], 'name': key[1], 'freq': freq}
        pf = frequencies(prime_factors(op['cost']))
        for name, factor in factor_map.items():
            op[name] = pf.pop(factor, 0)
        if pf and print_missed:
            import warnings
            warnings.warn(f'Missed prime factor(s) {pf} in cost scaling computation,  for operation {op}.')
        scalings.append(op)
    scalings.sort(key=lambda x: x['cost'], reverse=True)
    return scalings