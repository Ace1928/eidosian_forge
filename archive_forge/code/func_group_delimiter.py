import collections
def group_delimiter(self, value):
    self._validate_char(value)
    self._locale['group'] = value
    return self