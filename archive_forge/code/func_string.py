import os.path
import sys
import codecs
import argparse
from lark import Lark, Transformer, v_args
def string(self, s):
    return self._extra_rule(s)