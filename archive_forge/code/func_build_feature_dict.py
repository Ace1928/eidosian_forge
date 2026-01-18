import torch
import unicodedata
from collections import Counter
from parlai.core.build_data import modelzoo_path
def build_feature_dict(opt):
    """
    Make mapping of feature option to feature index.
    """
    feature_dict = {}
    if opt['use_in_question']:
        feature_dict['in_question'] = len(feature_dict)
        feature_dict['in_question_uncased'] = len(feature_dict)
    if opt['use_tf']:
        feature_dict['tf'] = len(feature_dict)
    if opt['use_time'] > 0:
        for i in range(opt['use_time'] - 1):
            feature_dict['time=T%d' % (i + 1)] = len(feature_dict)
        feature_dict['time>=T%d' % opt['use_time']] = len(feature_dict)
    return feature_dict