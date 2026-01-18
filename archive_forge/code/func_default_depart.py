import sys
from os.path import splitext
from docutils import parsers, nodes
from sphinx import addnodes
from commonmark import Parser
from warnings import warn
def default_depart(self, mdnode):
    """Default node depart handler

        If there is a matching ``visit_<type>`` method for a container node,
        then we should make sure to back up to it's parent element when the node
        is exited.
        """
    if mdnode.is_container():
        fn_name = 'visit_{0}'.format(mdnode.t)
        if not hasattr(self, fn_name):
            warn('Container node skipped: type={0}'.format(mdnode.t))
        else:
            self.current_node = self.current_node.parent