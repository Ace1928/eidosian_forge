import collections
import json
import os
import re
from typing import Optional, Tuple
import numpy as np
from ...tokenization_utils_fast import PreTrainedTokenizer
from ...utils import logging
def clean_text(self, content):
    content = self.content_repatter1.sub('<URL>', content)
    content = self.content_repatter2.sub('<EMAIL>', content)
    content = self.content_repatter3.sub('<TEL>', content)
    content = self.content_repatter4.sub('<DATE>', content)
    content = self.content_repatter5.sub('<DATE>', content)
    content = self.content_repatter6.sub('<PRICE>', content)
    content = content.translate(self.content_trans1)
    while '<BLOCK><BLOCK>' in content:
        content = content.replace('<BLOCK><BLOCK>', '<BLOCK>')
    return content