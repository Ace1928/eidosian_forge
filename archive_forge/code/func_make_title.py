import sys
import os.path
import csv
from docutils import io, nodes, statemachine, utils
from docutils.utils.error_reporting import SafeString
from docutils.utils import SystemMessagePropagation
from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives
def make_title(self):
    if self.arguments:
        title_text = self.arguments[0]
        text_nodes, messages = self.state.inline_text(title_text, self.lineno)
        title = nodes.title(title_text, '', *text_nodes)
        title.source, title.line = self.state_machine.get_source_and_line(self.lineno)
    else:
        title = None
        messages = []
    return (title, messages)