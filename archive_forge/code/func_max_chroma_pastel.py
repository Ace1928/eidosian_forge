import operator
import math
def max_chroma_pastel(L):
    H = math.degrees(_hrad_extremum(L))
    return max_chroma(L, H)