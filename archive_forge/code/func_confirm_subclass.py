import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
def confirm_subclass(self, value, state):
    if not isinstance(value, self.subclass):
        if len(self.subclass) == 1:
            msg = self.message('subclass', state, object=value, subclass=self.subclass[0])
        else:
            subclass_list = ', '.join(map(str, self.subclass))
            msg = self.message('inSubclass', state, object=value, subclassList=subclass_list)
        raise Invalid(msg, value, state)