import re
import itertools
@staticmethod
def glog(n):
    if n < 1:
        raise Exception('glog(' + n + ')')
    return LOG_TABLE[n]