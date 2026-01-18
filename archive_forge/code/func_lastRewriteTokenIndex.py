from io import StringIO
from antlr4.Token import Token
from antlr4.CommonTokenStream import CommonTokenStream
def lastRewriteTokenIndex(self, program_name=DEFAULT_PROGRAM_NAME):
    return self.lastRewriteTokenIndexes.get(program_name, -1)