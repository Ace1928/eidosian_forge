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
def add_to_ans_dict(self, token):
    """
        Builds dictionary from the list of provided tokens.

        Only adds words contained in self.embedding_words, if not None.
        """
    self.ansfreq[token] += 1
    if token not in self.ans2ind:
        index = len(self.ans2ind)
        self.ans2ind[token] = index
        self.ind2ans[index] = token