def rgb_to_hls(r, g, b):
    maxc = max(r, g, b)
    minc = min(r, g, b)
    sumc = maxc + minc
    rangec = maxc - minc
    l = sumc / 2.0
    if minc == maxc:
        return (0.0, l, 0.0)
    if l <= 0.5:
        s = rangec / sumc
    else:
        s = rangec / (2.0 - maxc - minc)
    rc = (maxc - r) / rangec
    gc = (maxc - g) / rangec
    bc = (maxc - b) / rangec
    if r == maxc:
        h = bc - gc
    elif g == maxc:
        h = 2.0 + rc - bc
    else:
        h = 4.0 + gc - rc
    h = h / 6.0 % 1.0
    return (h, l, s)