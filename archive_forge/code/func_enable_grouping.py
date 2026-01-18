from sqlparse import lexer
from sqlparse.engine import grouping
from sqlparse.engine.statement_splitter import StatementSplitter
def enable_grouping(self):
    self._grouping = True