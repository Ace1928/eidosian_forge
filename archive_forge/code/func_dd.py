import numpy as np
def dd(s1, s2, permute):
    if permute:
        s2 = s2.copy()
        dist = 0
        for a in s1:
            imin = None
            dmin = np.Inf
            for i, b in enumerate(s2):
                if a.symbol == b.symbol:
                    d = np.sum((a.position - b.position) ** 2)
                    if d < dmin:
                        dmin = d
                        imin = i
            dist += dmin
            s2.pop(imin)
        return np.sqrt(dist)
    else:
        return np.linalg.norm(s1.get_positions() - s2.get_positions())