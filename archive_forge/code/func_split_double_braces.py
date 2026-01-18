import sys
import re
def split_double_braces(input):
    """Masks out the quotes and comments, and then splits appropriate
  lines (lines that matche the double_*_brace re's above) before
  indenting them below.

  These are used to split lines which have multiple braces on them, so
  that the indentation looks prettier when all laid out (e.g. closing
  braces make a nice diagonal line).
  """
    double_open_brace_re = re.compile('(.*?[\\[\\{\\(,])(\\s*)([\\[\\{\\(])')
    double_close_brace_re = re.compile('(.*?[\\]\\}\\)],?)(\\s*)([\\]\\}\\)])')
    masked_input = mask_quotes(input)
    masked_input = mask_comments(masked_input)
    output, mask_output = do_split(input, masked_input, double_open_brace_re)
    output, mask_output = do_split(output, mask_output, double_close_brace_re)
    return output