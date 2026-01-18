import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def parse_structured_value(value):
    vs = value.lstrip()
    vs = value.replace(v[:len(value) - len(vs)], '\n')[1:]
    if vs.startswith('-'):
        r = []
        for match in re.findall(self._key_val_list_pat, vs):
            if match[0] and (not match[1]) and (not match[2]):
                r.append(match[0].strip())
            elif match[0] == '>' and (not match[1]) and match[2]:
                r.append(match[2].strip())
            elif match[0] and match[1]:
                r.append({match[0].strip(): match[1].strip()})
            elif not match[0] and (not match[1]) and match[2]:
                r.append(parse_structured_value(match[2]))
            else:
                pass
        return r
    else:
        return {match[0].strip(): match[1].strip() if match[1] else parse_structured_value(match[2]) for match in re.findall(self._key_val_dict_pat, vs)}