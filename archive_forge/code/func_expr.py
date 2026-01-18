import os.path
import sys
import codecs
import argparse
from lark import Lark, Transformer, v_args
def expr(self, item, op):
    rule = '(%s)%s' % (item, op)
    return self._extra_rule(rule)