from parlai.core.teachers import FixedDialogTeacher
from .build import build
import os
import json
def setup_helper(self, jsons_path):
    if self.opt['datatype'].startswith('test'):
        dpath = os.path.join(jsons_path, 'test.json')
    elif self.opt['datatype'].startswith('valid'):
        dpath = os.path.join(jsons_path, 'dev.json')
    elif self.opt['datatype'].startswith('train'):
        dpath = os.path.join(jsons_path, 'train.json')
    else:
        raise ValueError('Datatype not train, test, or valid')
    episodes = []
    with open(dpath) as f:
        data = json.load(f)
        for dialogue in data:
            context = '\n'.join(dialogue[0])
            qas = dialogue[1]
            episodes.append({'context': context, 'qas': qas})
    return episodes