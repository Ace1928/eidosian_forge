import codecs
import re
from mako import exceptions
from mako import parsetree
from mako.pygen import adjust_whitespace
def append_node(self, nodecls, *args, **kwargs):
    kwargs.setdefault('source', self.text)
    kwargs.setdefault('lineno', self.matched_lineno)
    kwargs.setdefault('pos', self.matched_charpos)
    kwargs['filename'] = self.filename
    node = nodecls(*args, **kwargs)
    if len(self.tag):
        self.tag[-1].nodes.append(node)
    else:
        self.template.nodes.append(node)
    if self.control_line:
        control_frame = self.control_line[-1]
        control_frame.nodes.append(node)
        if not (isinstance(node, parsetree.ControlLine) and control_frame.is_ternary(node.keyword)) and self.ternary_stack and self.ternary_stack[-1]:
            self.ternary_stack[-1][-1].nodes.append(node)
    if isinstance(node, parsetree.Tag):
        if len(self.tag):
            node.parent = self.tag[-1]
        self.tag.append(node)
    elif isinstance(node, parsetree.ControlLine):
        if node.isend:
            self.control_line.pop()
            self.ternary_stack.pop()
        elif node.is_primary:
            self.control_line.append(node)
            self.ternary_stack.append([])
        elif self.control_line and self.control_line[-1].is_ternary(node.keyword):
            self.ternary_stack[-1].append(node)
        elif self.control_line and (not self.control_line[-1].is_ternary(node.keyword)):
            raise exceptions.SyntaxException("Keyword '%s' not a legal ternary for keyword '%s'" % (node.keyword, self.control_line[-1].keyword), **self.exception_kwargs)