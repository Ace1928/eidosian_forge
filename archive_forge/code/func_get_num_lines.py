from ._base import *
@timed_cache(120)
def get_num_lines(self):
    return self.__len__