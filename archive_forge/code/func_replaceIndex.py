from io import StringIO
from antlr4.Token import Token
from antlr4.CommonTokenStream import CommonTokenStream
def replaceIndex(self, index, text):
    self.replace(self.DEFAULT_PROGRAM_NAME, index, index, text)