import re
def aggraget_str(str_children):
    return '\n'.join(['    ' + line for str_child in str_children for line in str_child.splitlines()])