import sys
import re
def prettyprint_input(lines):
    """Does the main work of indenting the input based on the brace counts."""
    indent = 0
    basic_offset = 2
    for line in lines:
        if COMMENT_RE.match(line):
            print(line)
        else:
            line = line.strip('\r\n\t ')
            if len(line) > 0:
                brace_diff, after = count_braces(line)
                if brace_diff != 0:
                    if after:
                        print(' ' * (basic_offset * indent) + line)
                        indent += brace_diff
                    else:
                        indent += brace_diff
                        print(' ' * (basic_offset * indent) + line)
                else:
                    print(' ' * (basic_offset * indent) + line)
            else:
                print('')