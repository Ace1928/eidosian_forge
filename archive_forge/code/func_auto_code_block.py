import os
import re
from docutils import nodes, transforms
from docutils.statemachine import StringList
from docutils.parsers.rst import Parser
from docutils.utils import new_document
from sphinx import addnodes
from .states import DummyStateMachine
def auto_code_block(self, node):
    """Try to automatically generate nodes for codeblock syntax.

        Parameters
        ----------
        node : nodes.literal_block
            Original codeblock node
        Returns
        -------
        tocnode: docutils node
            The converted toc tree node, None if conversion is not possible.
        """
    assert isinstance(node, nodes.literal_block)
    original_node = node
    if 'language' not in node:
        return None
    self.state_machine.reset(self.document, node.parent, self.current_level)
    content = node.rawsource.split('\n')
    language = node['language']
    if language == 'math':
        if self.config['enable_math']:
            return self.state_machine.run_directive('math', content=content)
    elif language == 'eval_rst':
        if self.config['enable_eval_rst']:
            node = nodes.section()
            self.state_machine.state.nested_parse(StringList(content, source=original_node.source), 0, node=node, match_titles=True)
            return node.children[:]
    else:
        match = re.search('[ ]?[\\w_-]+::.*', language)
        if match:
            parser = Parser()
            new_doc = new_document(None, self.document.settings)
            newsource = u'.. ' + match.group(0) + '\n' + node.rawsource
            parser.parse(newsource, new_doc)
            return new_doc.children[:]
        else:
            return self.state_machine.run_directive('code-block', arguments=[language], content=content)
    return None