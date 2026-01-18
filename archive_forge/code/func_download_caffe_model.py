import os
import argparse
from convert_model import convert_model
from convert_mean import convert_mean
import mxnet as mx
def download_caffe_model(model_name, meta_info, dst_dir='./model'):
    """Download caffe model into disk by the given meta info """
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    model_name = os.path.join(dst_dir, model_name)
    assert 'prototxt' in meta_info, 'missing prototxt url'
    proto_url, proto_sha1 = meta_info['prototxt']
    prototxt = mx.gluon.utils.download(proto_url, model_name + '_deploy.prototxt', sha1_hash=proto_sha1)
    assert 'caffemodel' in meta_info, 'mssing caffemodel url'
    caffemodel_url, caffemodel_sha1 = meta_info['caffemodel']
    caffemodel = mx.gluon.utils.download(caffemodel_url, model_name + '.caffemodel', sha1_hash=caffemodel_sha1)
    assert 'mean' in meta_info, 'no mean info'
    mean = meta_info['mean']
    if isinstance(mean[0], str):
        mean_url, mean_sha1 = mean
        mean = mx.gluon.utils.download(mean_url, model_name + '_mean.binaryproto', sha1_hash=mean_sha1)
    return (prototxt, caffemodel, mean)