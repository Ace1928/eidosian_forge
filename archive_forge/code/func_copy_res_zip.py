from builtins import zip
from builtins import str
import os
import os.path as op
import sys
from xml.etree import cElementTree as ET
import pyxnat
def copy_res_zip(src_r, dest_r, cache_d):
    """
    Copy a resource from XNAT source to XNAT destination using local cache
    in between
    """
    try:
        print('INFO:Downloading resource as zip...')
        cache_z = src_r.get(cache_d, extract=False)
        print('INFO:Uploading resource as zip...')
        dest_r.put_zip(cache_z, extract=True)
        os.remove(cache_z)
    except IndexError:
        print('ERROR:failed to copy:%s:%s' % (cache_z, sys.exc_info()[0]))
        raise