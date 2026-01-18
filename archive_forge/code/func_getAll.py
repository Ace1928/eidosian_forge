from io import StringIO
from antlr4.tree.ParseTreePattern import ParseTreePattern
from antlr4.tree.Tree import ParseTree
def getAll(self, label: str):
    nodes = self.labels.get(label, None)
    if nodes is None:
        return list()
    else:
        return nodes