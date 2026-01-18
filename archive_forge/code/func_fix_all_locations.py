from fontTools.varLib.models import VariationModel, normalizeValue, piecewiseLinearMap
def fix_all_locations(self):
    self.values = {Location(self.fix_location(l)): v for l, v in self.values.items()}