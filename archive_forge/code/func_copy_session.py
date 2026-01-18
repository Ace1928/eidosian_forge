from builtins import zip
from builtins import str
import os
import os.path as op
import sys
from xml.etree import cElementTree as ET
import pyxnat
def copy_session(src_sess, dst_sess, sess_cache_dir):
    """Copy XNAT session from source to destination"""
    print('INFO:uploading session attributes as xml')
    if not op.exists(sess_cache_dir):
        os.makedirs(sess_cache_dir)
    sess_xml = src_sess.get()
    xml_path = op.join(sess_cache_dir, 'sess.xml')
    write_xml(sess_xml, xml_path)
    sess_type = src_sess.datatype()
    dst_sess.create(experiments=sess_type)
    copy_attributes(src_sess, dst_sess)
    for src_scan in src_sess.scans().fetchall('obj'):
        scan_label = src_scan.label()
        print('INFO:Processing scan:%s...' % scan_label)
        dst_scan = dst_sess.scan(scan_label)
        scan_cache_dir = op.join(sess_cache_dir, scan_label)
        copy_scan(src_scan, dst_scan, scan_cache_dir)
    for src_assr in src_sess.assessors():
        assr_label = src_assr.label()
        print('INFO:Processing assessor:%s:...' % assr_label)
    for src_res in src_sess.resources().fetchall('obj'):
        res_label = src_res.label()
        print('INFO:Processing resource:%s...' % res_label)
        dst_res = dst_sess.resource(res_label)
        res_cache_dir = op.join(sess_cache_dir, res_label)
        copy_res(src_res, dst_res, res_cache_dir, use_zip=True)