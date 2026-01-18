from fontTools.misc.roundTools import noRound
from .errors import VariationModelError
def reorderMasters(self, master_list, mapping):
    new_list = [master_list[idx] for idx in mapping]
    self.origLocations = [self.origLocations[idx] for idx in mapping]
    locations = [{k: v for k, v in loc.items() if v != 0.0} for loc in self.origLocations]
    self.mapping = [self.locations.index(l) for l in locations]
    self.reverseMapping = [locations.index(l) for l in self.locations]
    self._subModels = {}
    return new_list