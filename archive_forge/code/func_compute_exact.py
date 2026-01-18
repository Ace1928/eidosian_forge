import collections
import json
import math
import re
import string
from ...models.bert import BasicTokenizer
from ...utils import logging
def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))