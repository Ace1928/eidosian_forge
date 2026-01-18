from lib2to3 import fixer_base
from lib2to3.fixer_util import LParen, RParen, Name
from libfuturize.fixer_util import touch_import_top
def insert_object(node, idx):
    node.insert_child(idx, RParen())
    node.insert_child(idx, Name(u'object'))
    node.insert_child(idx, LParen())