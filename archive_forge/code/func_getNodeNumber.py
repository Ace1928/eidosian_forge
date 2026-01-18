from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from antlr3.tree import CommonTreeAdaptor
from six.moves import range
import stringtemplate3
def getNodeNumber(self, t):
    try:
        return self.nodeToNumberMap[t]
    except KeyError:
        self.nodeToNumberMap[t] = self.nodeNumber
        self.nodeNumber += 1
        return self.nodeNumber - 1