import enum
import gast
def copyanno(from_node, to_node, key, field_name='___pyct_anno'):
    if hasanno(from_node, key, field_name=field_name):
        setanno(to_node, key, getanno(from_node, key, field_name=field_name), field_name=field_name)