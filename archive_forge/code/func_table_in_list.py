import re
from ..helpers import PREVENT_BACKSLASH
def table_in_list(md):
    """Enable table plugin in list."""
    md.block.insert_rule(md.block.list_rules, 'table', before='paragraph')
    md.block.insert_rule(md.block.list_rules, 'nptable', before='paragraph')