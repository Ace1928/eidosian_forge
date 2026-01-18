import re
from pygments.lexer import Lexer, RegexLexer, ExtendedRegexLexer, include, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import shebang_matches
def gen_rubystrings_rules():

    def intp_regex_callback(self, match, ctx):
        yield (match.start(1), String.Regex, match.group(1))
        nctx = LexerContext(match.group(3), 0, ['interpolated-regex'])
        for i, t, v in self.get_tokens_unprocessed(context=nctx):
            yield (match.start(3) + i, t, v)
        yield (match.start(4), String.Regex, match.group(4))
        ctx.pos = match.end()

    def intp_string_callback(self, match, ctx):
        yield (match.start(1), String.Other, match.group(1))
        nctx = LexerContext(match.group(3), 0, ['interpolated-string'])
        for i, t, v in self.get_tokens_unprocessed(context=nctx):
            yield (match.start(3) + i, t, v)
        yield (match.start(4), String.Other, match.group(4))
        ctx.pos = match.end()
    states = {}
    states['strings'] = [('\\:@{0,2}[a-zA-Z_]\\w*[!?]?', String.Symbol), (words(RUBY_OPERATORS, prefix='\\:@{0,2}'), String.Symbol), (":'(\\\\\\\\|\\\\'|[^'])*'", String.Symbol), ("'(\\\\\\\\|\\\\'|[^'])*'", String.Single), (':"', String.Symbol, 'simple-sym'), ('([a-zA-Z_]\\w*)(:)(?!:)', bygroups(String.Symbol, Punctuation)), ('"', String.Double, 'simple-string'), ('(?<!\\.)`', String.Backtick, 'simple-backtick')]
    for name, ttype, end in (('string', String.Double, '"'), ('sym', String.Symbol, '"'), ('backtick', String.Backtick, '`')):
        states['simple-' + name] = [include('string-intp-escaped'), ('[^\\\\%s#]+' % end, ttype), ('[\\\\#]', ttype), (end, ttype, '#pop')]
    for lbrace, rbrace, bracecc, name in (('\\{', '\\}', '{}', 'cb'), ('\\[', '\\]', '\\[\\]', 'sb'), ('\\(', '\\)', '()', 'pa'), ('<', '>', '<>', 'ab')):
        states[name + '-intp-string'] = [('\\\\[\\\\' + bracecc + ']', String.Other), (lbrace, String.Other, '#push'), (rbrace, String.Other, '#pop'), include('string-intp-escaped'), ('[\\\\#' + bracecc + ']', String.Other), ('[^\\\\#' + bracecc + ']+', String.Other)]
        states['strings'].append(('%[QWx]?' + lbrace, String.Other, name + '-intp-string'))
        states[name + '-string'] = [('\\\\[\\\\' + bracecc + ']', String.Other), (lbrace, String.Other, '#push'), (rbrace, String.Other, '#pop'), ('[\\\\#' + bracecc + ']', String.Other), ('[^\\\\#' + bracecc + ']+', String.Other)]
        states['strings'].append(('%[qsw]' + lbrace, String.Other, name + '-string'))
        states[name + '-regex'] = [('\\\\[\\\\' + bracecc + ']', String.Regex), (lbrace, String.Regex, '#push'), (rbrace + '[mixounse]*', String.Regex, '#pop'), include('string-intp'), ('[\\\\#' + bracecc + ']', String.Regex), ('[^\\\\#' + bracecc + ']+', String.Regex)]
        states['strings'].append(('%r' + lbrace, String.Regex, name + '-regex'))
    states['strings'] += [('(%r([\\W_]))((?:\\\\\\2|(?!\\2).)*)(\\2[mixounse]*)', intp_regex_callback), ('%[qsw]([\\W_])((?:\\\\\\1|(?!\\1).)*)\\1', String.Other), ('(%[QWx]([\\W_]))((?:\\\\\\2|(?!\\2).)*)(\\2)', intp_string_callback), ('(?<=[-+/*%=<>&!^|~,(])(\\s*)(%([\\t ])(?:(?:\\\\\\3|(?!\\3).)*)\\3)', bygroups(Text, String.Other, None)), ('^(\\s*)(%([\\t ])(?:(?:\\\\\\3|(?!\\3).)*)\\3)', bygroups(Text, String.Other, None)), ('(%([^a-zA-Z0-9\\s]))((?:\\\\\\2|(?!\\2).)*)(\\2)', intp_string_callback)]
    return states