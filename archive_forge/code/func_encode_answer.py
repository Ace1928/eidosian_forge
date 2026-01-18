import json
import os
import re
import numpy as np
from collections import Counter
from parlai.core.agents import Agent
from collections import defaultdict
from parlai.core.teachers import FixedDialogTeacher
from parlai.core.image_featurizers import ImageLoader
from .build import build
from parlai.tasks.coco_caption.build_2014 import buildImage as buildImage_2014
from parlai.tasks.coco_caption.build_2015 import buildImage as buildImage_2015
def encode_answer(self, examples):
    for ex in examples:
        if self.opt.get('samplingans', True):
            labels = ex.get('labels', ex.get('eval_labels'))
            ans_count = Counter(labels).most_common()
            valid_ans = []
            valid_count = []
            for ans in ans_count:
                if ans[0] in self.ans2ind:
                    valid_ans.append(self.ans2ind[ans[0]])
                    valid_count.append(ans[1])
            if not valid_ans:
                ex['answer_aid'] = 0
            else:
                probs = valid_count / np.sum(valid_count)
                ex['answer_aid'] = int(np.random.choice(valid_ans, p=probs))
        else:
            ex['answer_aid'] = self.ans2ind[ex['mc_label'][0]]
    return examples