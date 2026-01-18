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
def run_directive(self, directive, match, type_name, option_presets):
    """
        Parse a directive then run its directive function.

        Parameters:

        - `directive`: The class implementing the directive.  Must be
          a subclass of `rst.Directive`.

        - `match`: A regular expression match object which matched the first
          line of the directive.

        - `type_name`: The directive name, as used in the source text.

        - `option_presets`: A dictionary of preset options, defaults for the
          directive options.  Currently, only an "alt" option is passed by
          substitution definitions (value: the substitution name), which may
          be used by an embedded image directive.

        Returns a 2-tuple: list of nodes, and a "blank finish" boolean.
        """
    if isinstance(directive, (FunctionType, MethodType)):
        from docutils.parsers.rst import convert_directive_function
        directive = convert_directive_function(directive)
    lineno = self.state_machine.abs_line_number()
    initial_line_offset = self.state_machine.line_offset
    indented, indent, line_offset, blank_finish = self.state_machine.get_first_known_indented(match.end(), strip_top=0)
    block_text = '\n'.join(self.state_machine.input_lines[initial_line_offset:self.state_machine.line_offset + 1])
    try:
        arguments, options, content, content_offset = self.parse_directive_block(indented, line_offset, directive, option_presets)
    except MarkupError as detail:
        error = self.reporter.error('Error in "%s" directive:\n%s.' % (type_name, ' '.join(detail.args)), nodes.literal_block(block_text, block_text), line=lineno)
        return ([error], blank_finish)
    directive_instance = directive(type_name, arguments, options, content, lineno, content_offset, block_text, self, self.state_machine)
    try:
        result = directive_instance.run()
    except docutils.parsers.rst.DirectiveError as error:
        msg_node = self.reporter.system_message(error.level, error.msg, line=lineno)
        msg_node += nodes.literal_block(block_text, block_text)
        result = [msg_node]
    assert isinstance(result, list), 'Directive "%s" must return a list of nodes.' % type_name
    for i in range(len(result)):
        assert isinstance(result[i], nodes.Node), 'Directive "%s" returned non-Node object (index %s): %r' % (type_name, i, result[i])
    return (result, blank_finish or self.state_machine.is_next_line_blank())