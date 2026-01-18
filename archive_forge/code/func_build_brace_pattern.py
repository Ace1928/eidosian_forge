from __future__ import print_function, absolute_import
from textwrap import dedent
def build_brace_pattern(level, separators=''):
    """
    Build a brace pattern upto a given depth

    The brace pattern parses any pattern with round, square, curly, or angle
    brackets. Inside those brackets, any characters are allowed.

    The structure is quite simple and is recursively repeated as needed.
    When separators are given, the match stops at that separator.

    Reason to use this instead of some Python function:
    The resulting regex is _very_ fast!

    A faster replacement would be written in C, but this solution is
    sufficient when the nesting level is not too large.

    Because of the recursive nature of the pattern, the size grows by a factor
    of 4 at every level, as does the creation time. Up to a level of 6, this
    is below 10 ms.

    There are other regex engines available which allow recursive patterns,
    avoiding this problem completely. It might be considered to switch to
    such an engine if the external module is not a problem.
    """

    def escape(str):
        return ''.join(('\\' + c for c in str))
    ro, rc = round = '()'
    so, sc = square = '[]'
    co, cc = curly = 'CD'
    ao, ac = angle = '<>'
    qu, bs = ('"', '\\')
    all = round + square + curly + angle
    __ = '  '
    ro, rc, so, sc, co, cc, ao, ac, separators, qu, bs, all = map(escape, (ro, rc, so, sc, co, cc, ao, ac, separators, qu, bs, all))
    no_brace_sep_q = '[^{all}{separators}{qu}{bs}]'.format(**locals())
    no_quote = '(?: [^{qu}{bs}] | {bs}. )*'.format(**locals())
    pattern = dedent('\n        (\n          (?: {__} {no_brace_sep_q}\n            | {qu} {no_quote} {qu}\n            | {ro} {replacer} {rc}\n            | {so} {replacer} {sc}\n            | {co} {replacer} {cc}\n            | {ao} {replacer} {ac}\n          )*\n        )\n        ')
    no_braces_q = '[^{all}{qu}{bs}]*'.format(**locals())
    repeated = dedent('\n        {indent}  (?: {__} {no_braces_q}\n        {indent}    | {qu} {no_quote} {qu}\n        {indent}    | {ro} {replacer} {rc}\n        {indent}    | {so} {replacer} {sc}\n        {indent}    | {co} {replacer} {cc}\n        {indent}    | {ao} {replacer} {ac}\n        {indent}  )*\n        ')
    for idx in range(level):
        pattern = pattern.format(replacer=repeated if idx < level - 1 else no_braces_q, indent=idx * '    ', **locals())
    return pattern.replace('C', '{').replace('D', '}')