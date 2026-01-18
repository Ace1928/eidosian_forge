from suds import *
from suds.sudsobject import Factory
def skip_value(self, type):
    """ whether or not to skip setting the value """
    return type.optional() and (not type.multi_occurrence())