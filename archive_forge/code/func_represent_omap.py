from __future__ import print_function, absolute_import, division
from ruamel.yaml.error import *  # NOQA
from ruamel.yaml.nodes import *  # NOQA
from ruamel.yaml.compat import text_type, binary_type, to_unicode, PY2, PY3, ordereddict
from ruamel.yaml.compat import nprint, nprintf  # NOQA
from ruamel.yaml.scalarstring import (
from ruamel.yaml.scalarint import ScalarInt, BinaryInt, OctalInt, HexInt, HexCapsInt
from ruamel.yaml.scalarfloat import ScalarFloat
from ruamel.yaml.scalarbool import ScalarBoolean
from ruamel.yaml.timestamp import TimeStamp
import datetime
import sys
import types
from ruamel.yaml.comments import (
def represent_omap(self, tag, omap, flow_style=None):
    value = []
    try:
        flow_style = omap.fa.flow_style(flow_style)
    except AttributeError:
        flow_style = flow_style
    try:
        anchor = omap.yaml_anchor()
    except AttributeError:
        anchor = None
    node = SequenceNode(tag, value, flow_style=flow_style, anchor=anchor)
    if self.alias_key is not None:
        self.represented_objects[self.alias_key] = node
    best_style = True
    try:
        comment = getattr(omap, comment_attrib)
        node.comment = comment.comment
        if node.comment and node.comment[1]:
            for ct in node.comment[1]:
                ct.reset()
        item_comments = comment.items
        for v in item_comments.values():
            if v and v[1]:
                for ct in v[1]:
                    ct.reset()
        try:
            node.comment.append(comment.end)
        except AttributeError:
            pass
    except AttributeError:
        item_comments = {}
    for item_key in omap:
        item_val = omap[item_key]
        node_item = self.represent_data({item_key: item_val})
        item_comment = item_comments.get(item_key)
        if item_comment:
            if item_comment[1]:
                node_item.comment = [None, item_comment[1]]
            assert getattr(node_item.value[0][0], 'comment', None) is None
            node_item.value[0][0].comment = [item_comment[0], None]
            nvc = getattr(node_item.value[0][1], 'comment', None)
            if nvc is not None:
                nvc[0] = item_comment[2]
                nvc[1] = item_comment[3]
            else:
                node_item.value[0][1].comment = item_comment[2:]
        value.append(node_item)
    if flow_style is None:
        if self.default_flow_style is not None:
            node.flow_style = self.default_flow_style
        else:
            node.flow_style = best_style
    return node