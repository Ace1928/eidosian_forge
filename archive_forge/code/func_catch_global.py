from numba import int32
from numba.typed import List
def catch_global():
    x = List()
    for i in global_typed_list:
        x.append(i)