import operator
import math
def max_chroma(L, H):
    hrad = math.radians(H)
    sinH = math.sin(hrad)
    cosH = math.cos(hrad)
    sub1 = math.pow(L + 16, 3.0) / 1560896.0
    sub2 = sub1 if sub1 > 0.008856 else L / 903.3
    result = float('inf')
    for row in m:
        m1 = row[0]
        m2 = row[1]
        m3 = row[2]
        top = (0.99915 * m1 + 1.05122 * m2 + 1.1446 * m3) * sub2
        rbottom = 0.8633 * m3 - 0.17266 * m2
        lbottom = 0.12949 * m3 - 0.38848 * m1
        bottom = (rbottom * sinH + lbottom * cosH) * sub2
        for t in (0.0, 1.0):
            C = L * (top - 1.05122 * t) / (bottom + 0.17266 * sinH * t)
            if C > 0.0 and C < result:
                result = C
    return result