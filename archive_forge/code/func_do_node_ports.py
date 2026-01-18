from pyparsing import (
import pydot
def do_node_ports(node):
    node_port = ''
    if len(node) > 1:
        node_port = ''.join([str(a) + str(b) for a, b in node[1]])
    return node_port