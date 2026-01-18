import collections
def padding_width(self, value):
    self._validate_non_negative_integer_or_none(value)
    self._specifier['width'] = value if value is not None else ''
    return self