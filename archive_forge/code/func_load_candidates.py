from parlai.core.teachers import FixedDialogTeacher
from parlai.core.image_featurizers import ImageLoader
from .build_2014 import build as build_2014
from .build_2014 import buildImage as buildImage_2014
from .build_2017 import build as build_2017
from .build_2017 import buildImage as buildImage_2017
import os
import json
import random
def load_candidates(datapath, datatype, version):
    if not datatype.startswith('train'):
        suffix = 'captions_{}{}.json'
        suffix_val = suffix.format('val', version)
        val_path = os.path.join(datapath, 'COCO_{}_Caption'.format(version), 'annotations', suffix_val)
        val = json.load(open(val_path))['annotations']
        val_caps = [x['caption'] for x in val]
        if datatype.startswith('test'):
            suffix_train = suffix.format('train', version)
            train_path = os.path.join(datapath, 'COCO_{}_Caption'.format(version), 'annotations', suffix_train)
            train = json.load(open(train_path))['annotations']
            train_caps = [x['caption'] for x in train]
            test_caps = train_caps + val_caps
            return test_caps
        else:
            return val_caps
    else:
        return None