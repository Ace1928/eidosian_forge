import os
import time
def str_tdelta(delt):
    if delt is None:
        return '-:--:--'
    delt = int(round(delt))
    return '%d:%02d:%02d' % (delt / 3600, delt / 60 % 60, delt % 60)