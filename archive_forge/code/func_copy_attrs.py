from builtins import zip
from builtins import str
import os
import os.path as op
import sys
from xml.etree import cElementTree as ET
import pyxnat
def copy_attrs(src_obj, dest_obj, attr_list):
    """ Copies list of attributes form source to destination"""
    src_attrs = src_obj.attrs.mget(attr_list)
    src_list = dict(list(zip(attr_list, src_attrs)))
    te_key = 'xnat:mrScanData/parameters/te'
    if te_key in src_list:
        src_list[te_key] = src_obj.attrs.get(te_key)
    dest_obj.attrs.mset(src_list)
    return 0