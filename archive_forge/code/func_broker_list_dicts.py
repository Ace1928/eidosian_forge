import sys
import os
import os.path
import ctypes
from ctypes import c_char_p, c_int, c_size_t, c_void_p, pointer, CFUNCTYPE, POINTER
import ctypes.util
import platform
import textwrap
def broker_list_dicts(broker, cbfunc):

    def cbfunc1(*args):
        cbfunc(*args[:-1])
    broker_list_dicts1(broker, t_dict_desc_func(cbfunc1), None)