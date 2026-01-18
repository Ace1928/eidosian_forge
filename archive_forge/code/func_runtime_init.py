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
def runtime_init(self):
    StateWS.runtime_init(self)
    memo = self.state_machine.memo
    self.memo = memo
    self.reporter = memo.reporter
    self.inliner = memo.inliner
    self.document = memo.document
    self.parent = self.state_machine.node
    if not hasattr(self.reporter, 'get_source_and_line'):
        self.reporter.get_source_and_line = self.state_machine.get_source_and_line