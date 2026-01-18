import enum
import gast
def hasanno(node, key, field_name='___pyct_anno'):
    return hasattr(node, field_name) and key in getattr(node, field_name)