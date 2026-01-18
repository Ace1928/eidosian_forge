import sys
def make_param(name):
    value = getattr(self, name)
    return '{name}={value!r}'.format(**locals())