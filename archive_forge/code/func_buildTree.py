import traceback
import types
from collections import OrderedDict
import numpy as np
from ..Qt import QtWidgets
from .TableWidget import TableWidget
def buildTree(self, data, parent, name='', hideRoot=False, path=()):
    if hideRoot:
        node = parent
    else:
        node = QtWidgets.QTreeWidgetItem([name, '', ''])
        parent.addChild(node)
    self.nodes[path] = node
    typeStr, desc, childs, widget = self.parse(data)
    if len(desc) > 100:
        desc = desc[:97] + '...'
        if widget is None:
            widget = QtWidgets.QPlainTextEdit(str(data))
            widget.setMaximumHeight(200)
            widget.setReadOnly(True)
    node.setText(1, typeStr)
    node.setText(2, desc)
    if widget is not None:
        self.widgets.append(widget)
        subnode = QtWidgets.QTreeWidgetItem(['', '', ''])
        node.addChild(subnode)
        self.setItemWidget(subnode, 0, widget)
        subnode.setFirstColumnSpanned(True)
    for key, data in childs.items():
        self.buildTree(data, node, str(key), path=path + (key,))