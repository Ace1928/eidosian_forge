import inspect
import re
import six
def sorting_keys(s):
    m = re.search('(.*?)(\\d+$)', str(s))
    if m:
        return (m.group(1), int(m.group(2)))
    else:
        return (str(s), 0)