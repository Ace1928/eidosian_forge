import sys
import re
def count_braces(line):
    """keeps track of the number of braces on a given line and returns the result.

  It starts at zero and subtracts for closed braces, and adds for open braces.
  """
    open_braces = ['[', '(', '{']
    close_braces = [']', ')', '}']
    closing_prefix_re = re.compile('(.*?[^\\s\\]\\}\\)]+.*?)([\\]\\}\\)],?)\\s*$')
    cnt = 0
    stripline = COMMENT_RE.sub('', line)
    stripline = QUOTE_RE.sub("''", stripline)
    for char in stripline:
        for brace in open_braces:
            if char == brace:
                cnt += 1
        for brace in close_braces:
            if char == brace:
                cnt -= 1
    after = False
    if cnt > 0:
        after = True
    if cnt < 0 and closing_prefix_re.match(stripline):
        after = True
    return (cnt, after)