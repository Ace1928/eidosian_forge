import os.path
import sys
import codecs
import argparse
from lark import Lark, Transformer, v_args
def ruledef(self, name, exps):
    return '!%s: %s' % (_get_rulename(name), exps)