from __future__ import absolute_import, division, unicode_literals
from six import text_type
from collections import OrderedDict
from lxml import etree
from ..treebuilders.etree import tag_regexp
from . import base
from .. import _ihatexml
def getNodeDetails(self, node):
    if isinstance(node, tuple):
        node, key = node
        assert key in ('text', 'tail'), 'Text nodes are text or tail, found %s' % key
        return (base.TEXT, ensure_str(getattr(node, key)))
    elif isinstance(node, Root):
        return (base.DOCUMENT,)
    elif isinstance(node, Doctype):
        return (base.DOCTYPE, node.name, node.public_id, node.system_id)
    elif isinstance(node, FragmentWrapper) and (not hasattr(node, 'tag')):
        return (base.TEXT, ensure_str(node.obj))
    elif node.tag == etree.Comment:
        return (base.COMMENT, ensure_str(node.text))
    elif node.tag == etree.Entity:
        return (base.ENTITY, ensure_str(node.text)[1:-1])
    else:
        match = tag_regexp.match(ensure_str(node.tag))
        if match:
            namespace, tag = match.groups()
        else:
            namespace = None
            tag = ensure_str(node.tag)
        attrs = OrderedDict()
        for name, value in list(node.attrib.items()):
            name = ensure_str(name)
            value = ensure_str(value)
            match = tag_regexp.match(name)
            if match:
                attrs[match.group(1), match.group(2)] = value
            else:
                attrs[None, name] = value
        return (base.ELEMENT, namespace, self.filter.fromXmlName(tag), attrs, len(node) > 0 or node.text)