import re
from threading import Lock
from io import TextIOBase
from sqlparse import tokens, keywords
from sqlparse.utils import consume
def default_initialization(self):
    """Initialize the lexer with default dictionaries.
        Useful if you need to revert custom syntax settings."""
    self.clear()
    self.set_SQL_REGEX(keywords.SQL_REGEX)
    self.add_keywords(keywords.KEYWORDS_COMMON)
    self.add_keywords(keywords.KEYWORDS_ORACLE)
    self.add_keywords(keywords.KEYWORDS_MYSQL)
    self.add_keywords(keywords.KEYWORDS_PLPGSQL)
    self.add_keywords(keywords.KEYWORDS_HQL)
    self.add_keywords(keywords.KEYWORDS_MSACCESS)
    self.add_keywords(keywords.KEYWORDS_SNOWFLAKE)
    self.add_keywords(keywords.KEYWORDS_BIGQUERY)
    self.add_keywords(keywords.KEYWORDS)