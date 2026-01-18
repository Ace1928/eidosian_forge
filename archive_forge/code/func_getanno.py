import enum
import gast
def getanno(node, key, default=FAIL, field_name='___pyct_anno'):
    if default is FAIL or (hasattr(node, field_name) and key in getattr(node, field_name)):
        return getattr(node, field_name)[key]
    return default