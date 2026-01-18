from ._base import *
@property
def as_strings(self):
    """Return key as nginx config string."""
    if self.value == '' or self.value is None:
        return '{0};\n'.format(self.name)
    if type(self.value) == str and '"' not in self.value and (';' in self.value or '#' in self.value):
        return '{0} "{1}";\n'.format(self.name, self.value)
    return '{0} {1};\n'.format(self.name, self.value)