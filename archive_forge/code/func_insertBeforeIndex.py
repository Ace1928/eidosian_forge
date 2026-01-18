from io import StringIO
from antlr4.Token import Token
from antlr4.CommonTokenStream import CommonTokenStream
def insertBeforeIndex(self, index, text):
    self.insertBefore(self.DEFAULT_PROGRAM_NAME, index, text)