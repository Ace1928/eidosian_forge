import os
import sys
import re
def resolve_includes(source):
    d = os.path.dirname(source)
    with open(source) as fid:
        lines = []
        for line in fid:
            m = include_src_re.match(line)
            if m:
                fn = m.group('name')
                if not os.path.isabs(fn):
                    fn = os.path.join(d, fn)
                if os.path.isfile(fn):
                    lines.extend(resolve_includes(fn))
                else:
                    lines.append(line)
            else:
                lines.append(line)
    return lines