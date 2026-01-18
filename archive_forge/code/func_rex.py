import re
def rex(expr):
    """ Regular expression matcher to use together with transform functions """
    r = re.compile(expr)
    return lambda key: isinstance(key, str) and r.match(key)