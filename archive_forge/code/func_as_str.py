import numbers as _numbers
import numpy as _np
import six as _six
import codecs
from tensorflow.python.util.tf_export import tf_export
def as_str(bytes_or_text, encoding='utf-8'):
    return as_text(bytes_or_text, encoding)