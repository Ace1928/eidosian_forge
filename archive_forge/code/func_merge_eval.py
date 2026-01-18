import collections
import json
import math
import re
import string
from ...models.bert import BasicTokenizer
from ...utils import logging
def merge_eval(main_eval, new_eval, prefix):
    for k in new_eval:
        main_eval[f'{prefix}_{k}'] = new_eval[k]