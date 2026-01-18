from io import StringIO
from antlr4.Token import Token
from antlr4.CommonTokenStream import CommonTokenStream
def replaceRangeTokens(self, from_token, to_token, text, program_name=DEFAULT_PROGRAM_NAME):
    self.replace(program_name, from_token.tokenIndex, to_token.tokenIndex, text)