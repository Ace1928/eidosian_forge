from io import StringIO
from antlr4.Token import Token
from antlr4.CommonTokenStream import CommonTokenStream
def deleteIndex(self, index):
    self.delete(self.DEFAULT_PROGRAM_NAME, index, index)