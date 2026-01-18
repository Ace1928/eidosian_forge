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
def add_to_ques_dict(self, tokens):
    """
        Builds dictionary from the list of provided tokens.

        Only adds words contained in self.embedding_words, if not None.
        """
    for token in tokens:
        self.freq[token] += 1
        if token not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[token] = index
            self.ind2tok[index] = token