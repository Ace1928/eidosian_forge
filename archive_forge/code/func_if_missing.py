from .api import (
@property
def if_missing(self):
    for validator in self.validators:
        v = validator.if_missing
        if v is not NoDefault:
            return v
    return NoDefault