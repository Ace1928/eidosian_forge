import enum
import gast
def delanno(node, key, field_name='___pyct_anno'):
    annotations = getattr(node, field_name)
    del annotations[key]
    if not annotations:
        delattr(node, field_name)
        node._fields = tuple((f for f in node._fields if f != field_name))