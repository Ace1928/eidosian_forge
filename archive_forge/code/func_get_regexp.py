import re
def get_regexp(self, pattern, match_from=False):
    result = None
    if isinstance(pattern, self.__six.string_types) and pattern != '':
        result = re.compile(pattern)
    elif pattern is not None:
        result = re.compile(pattern.pattern)
    return result