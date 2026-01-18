import ast
import tokenize
from argparse import ArgumentParser, RawTextHelpFormatter
from enum import Enum
from nodedump import debug_format_node
def parse_ast(filename):
    node = None
    with tokenize.open(filename) as f:
        global _source_lines
        source = f.read()
        _source_lines = source.split('\n')
        node = ast.parse(source, mode='exec')
    return node