from suds import *
from suds.sudsobject import Factory
def skip_child(self, child, ancestry):
    """ get whether or not to skip the specified child """
    if child.any():
        return True
    for x in ancestry:
        if x.choice():
            return True
    return False