from __future__ import unicode_literals
@classmethod
def assign_names(cls):
    for key, value in vars(cls).items():
        if isinstance(value, cls):
            value.name_ = key