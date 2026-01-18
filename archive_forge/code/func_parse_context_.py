import fontTools.voltLib.ast as ast
from fontTools.voltLib.lexer import Lexer
from fontTools.voltLib.error import VoltLibError
from io import open
def parse_context_(self):
    location = self.cur_token_location_
    contexts = []
    while self.next_token_ in ('EXCEPT_CONTEXT', 'IN_CONTEXT'):
        side = None
        coverage = None
        ex_or_in = self.expect_name_()
        if self.next_token_ != 'END_CONTEXT':
            left = []
            right = []
            while self.next_token_ in ('LEFT', 'RIGHT'):
                side = self.expect_name_()
                coverage = self.parse_coverage_()
                if side == 'LEFT':
                    left.append(coverage)
                else:
                    right.append(coverage)
            self.expect_keyword_('END_CONTEXT')
            context = ast.ContextDefinition(ex_or_in, left, right, location=location)
            contexts.append(context)
        else:
            self.expect_keyword_('END_CONTEXT')
    return contexts