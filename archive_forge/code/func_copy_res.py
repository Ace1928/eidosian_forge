from builtins import zip
from builtins import str
import os
import os.path as op
import sys
from xml.etree import cElementTree as ET
import pyxnat
def copy_res(src_res, dst_res, res_cache_dir, use_zip=False):
    """Copy resource from source XNAT to destination XNAT"""
    if not op.exists(res_cache_dir):
        os.makedirs(res_cache_dir)
    is_empty = False
    print(dst_res._uri)
    if not dst_res.exists():
        dst_res.create()
        is_empty = True
    elif is_empty_resource(dst_res):
        is_empty = True
    if is_empty_resource(src_res):
        print('WARN:empty resource, nothing to copy')
        return
    if is_empty:
        if use_zip:
            try:
                print('INFO:Copying resource as zip: %s...' % src_res.label())
                copy_res_zip(src_res, dst_res, res_cache_dir)
                return
            except Exception:
                try:
                    print('INFO: second attempt to copy resource as zip: %s...' % src_res.label())
                    copy_res_zip(src_res, dst_res, res_cache_dir)
                    return
                except Exception:
                    msg = 'ERROR:failed twice to copy resource as zip, willcopy individual files'
                    print(msg)
        copy_count = 0
        for f in src_res.files():
            print('INFO:Copying file: %s...' % f.label())
            copy_count += 1
            copy_file(f, dst_res, res_cache_dir)
        print('INFO:Finished copying resource, %d files copied' % copy_count)