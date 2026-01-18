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
def encode_question(self, examples, training):
    minwcount = self.opt.get('minwcount', 0)
    maxlength = self.opt.get('maxlength', 16)
    for ex in examples:
        words = self.tokenize_mcb(ex['text'])
        if training:
            words_unk = [w if self.freq.get(w, 0) > minwcount else self.unk_token for w in words]
        else:
            words_unk = [w if w in self.tok2ind else self.unk_token for w in words]
        ex['question_wids'] = [self.tok2ind[self.null_token]] * maxlength
        for k, w in enumerate(words_unk):
            if k < maxlength:
                ex['question_wids'][k] = self.tok2ind[w]
    return examples