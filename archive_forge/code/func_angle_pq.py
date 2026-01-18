from kivy.vector import Vector
def angle_pq(r):
    if r in (P, Q):
        return 10000000000.0
    return abs((r - P).angle(r - Q))