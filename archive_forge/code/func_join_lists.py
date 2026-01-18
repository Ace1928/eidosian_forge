import string
from ..sage_helper import _within_sage, sage_method
def join_lists(list_of_lists):
    for L in list_of_lists:
        yield from L