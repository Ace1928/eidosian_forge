import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
def convert_string_to_list(string_val):
    """Helper function to convert string to list.
     Used to convert shape attribute string to list format.
    """
    result_list = []
    list_string = string_val.split(',')
    for val in list_string:
        val = str(val.strip())
        val = val.replace('(', '')
        val = val.replace(')', '')
        val = val.replace('L', '')
        val = val.replace('[', '')
        val = val.replace(']', '')
        if val == 'None':
            result_list.append(None)
        elif val != '':
            result_list.append(int(val))
    return result_list