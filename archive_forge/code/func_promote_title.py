import re
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
def promote_title(self, node):
    """
        Transform the following tree::

            <node>
                <section>
                    <title>
                    ...

        into ::

            <node>
                <title>
                ...

        `node` is normally a document.
        """
    if not isinstance(node, nodes.Element):
        raise TypeError('node must be of Element-derived type.')
    assert not (len(node) and isinstance(node[0], nodes.title))
    section, index = self.candidate_index(node)
    if index is None:
        return None
    node.update_all_atts_concatenating(section, True, True)
    node[:] = section[:1] + node[:index] + section[1:]
    assert isinstance(node[0], nodes.title)
    return 1