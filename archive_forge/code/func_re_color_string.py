import os
import sys
def re_color_string(compiled_pattern, s, fg):
    return compiled_pattern.sub(fg + '\\1' + FG.NONE, s)