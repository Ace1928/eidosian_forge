import re
from pygments.lexer import RegexLexer, ExtendedRegexLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import shebang_matches
def brackets_callback(token_class):

    def callback(lexer, match, context):
        groups = match.groupdict()
        opening_chars = groups['delimiter']
        n_chars = len(opening_chars)
        adverbs = groups.get('adverbs')
        closer = Perl6Lexer.PERL6_BRACKETS.get(opening_chars[0])
        text = context.text
        if closer is None:
            end_pos = text.find(opening_chars, match.start('delimiter') + n_chars)
        else:
            closing_chars = closer * n_chars
            nesting_level = 1
            search_pos = match.start('delimiter')
            while nesting_level > 0:
                next_open_pos = text.find(opening_chars, search_pos + n_chars)
                next_close_pos = text.find(closing_chars, search_pos + n_chars)
                if next_close_pos == -1:
                    next_close_pos = len(text)
                    nesting_level = 0
                elif next_open_pos != -1 and next_open_pos < next_close_pos:
                    nesting_level += 1
                    search_pos = next_open_pos
                else:
                    nesting_level -= 1
                    search_pos = next_close_pos
            end_pos = next_close_pos
        if end_pos < 0:
            end_pos = len(text)
        if adverbs is not None and re.search(':to\\b', adverbs):
            heredoc_terminator = text[match.start('delimiter') + n_chars:end_pos]
            end_heredoc = re.search('^\\s*' + re.escape(heredoc_terminator) + '\\s*$', text[end_pos:], re.MULTILINE)
            if end_heredoc:
                end_pos += end_heredoc.end()
            else:
                end_pos = len(text)
        yield (match.start(), token_class, text[match.start():end_pos + n_chars])
        context.pos = end_pos + n_chars
    return callback