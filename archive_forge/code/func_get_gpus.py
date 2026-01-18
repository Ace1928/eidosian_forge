from measure import run
import subprocess
import logging
import mxnet as mx
def get_gpus():
    return ','.join([str(i) for i in range(mx.util.get_gpu_count())])