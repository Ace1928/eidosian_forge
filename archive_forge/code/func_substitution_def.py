import sys
import re
from types import FunctionType, MethodType
from docutils import nodes, statemachine, utils
from docutils import ApplicationError, DataError
from docutils.statemachine import StateMachineWS, StateWS
from docutils.nodes import fully_normalize_name as normalize_name
from docutils.nodes import whitespace_normalize_name
import docutils.parsers.rst
from docutils.parsers.rst import directives, languages, tableparser, roles
from docutils.parsers.rst.languages import en as _fallback_language_module
from docutils.utils import escape2null, unescape, column_width
from docutils.utils import punctuation_chars, roman, urischemes
from docutils.utils import split_escaped_whitespace
def substitution_def(self, match):
    pattern = self.explicit.patterns.substitution
    src, srcline = self.state_machine.get_source_and_line()
    block, indent, offset, blank_finish = self.state_machine.get_first_known_indented(match.end(), strip_indent=False)
    blocktext = match.string[:match.end()] + '\n'.join(block)
    block.disconnect()
    escaped = escape2null(block[0].rstrip())
    blockindex = 0
    while True:
        subdefmatch = pattern.match(escaped)
        if subdefmatch:
            break
        blockindex += 1
        try:
            escaped = escaped + ' ' + escape2null(block[blockindex].strip())
        except IndexError:
            raise MarkupError('malformed substitution definition.')
    del block[:blockindex]
    block[0] = (block[0].strip() + ' ')[subdefmatch.end() - len(escaped) - 1:-1]
    if not block[0]:
        del block[0]
        offset += 1
    while block and (not block[-1].strip()):
        block.pop()
    subname = subdefmatch.group('name')
    substitution_node = nodes.substitution_definition(blocktext)
    substitution_node.source = src
    substitution_node.line = srcline
    if not block:
        msg = self.reporter.warning('Substitution definition "%s" missing contents.' % subname, nodes.literal_block(blocktext, blocktext), source=src, line=srcline)
        return ([msg], blank_finish)
    block[0] = block[0].strip()
    substitution_node['names'].append(nodes.whitespace_normalize_name(subname))
    new_abs_offset, blank_finish = self.nested_list_parse(block, input_offset=offset, node=substitution_node, initial_state='SubstitutionDef', blank_finish=blank_finish)
    i = 0
    for node in substitution_node[:]:
        if not (isinstance(node, nodes.Inline) or isinstance(node, nodes.Text)):
            self.parent += substitution_node[i]
            del substitution_node[i]
        else:
            i += 1
    for node in substitution_node.traverse(nodes.Element):
        if self.disallowed_inside_substitution_definitions(node):
            pformat = nodes.literal_block('', node.pformat().rstrip())
            msg = self.reporter.error('Substitution definition contains illegal element <%s>:' % node.tagname, pformat, nodes.literal_block(blocktext, blocktext), source=src, line=srcline)
            return ([msg], blank_finish)
    if len(substitution_node) == 0:
        msg = self.reporter.warning('Substitution definition "%s" empty or invalid.' % subname, nodes.literal_block(blocktext, blocktext), source=src, line=srcline)
        return ([msg], blank_finish)
    self.document.note_substitution_def(substitution_node, subname, self.parent)
    return ([substitution_node], blank_finish)