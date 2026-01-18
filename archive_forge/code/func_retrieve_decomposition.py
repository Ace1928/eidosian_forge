from . import matrix
from . import homology
from .ptolemyObstructionClass import PtolemyObstructionClass
from .ptolemyGeneralizedObstructionClass import PtolemyGeneralizedObstructionClass
from .ptolemyVariety import PtolemyVariety
from .utilities import MethodMappingList
def retrieve_decomposition(self, *args, **kwargs):
    return MethodMappingList([p.retrieve_decomposition(*args, **kwargs) for p in self])