from fontTools.varLib.models import VariationModel, normalizeValue, piecewiseLinearMap
def add_to_variation_store(self, store_builder, model_cache=None, avar=None):
    deltas, supports = self.get_deltas_and_supports(model_cache, avar)
    store_builder.setSupports(supports)
    index = store_builder.storeDeltas(deltas)
    return (int(self.default), index)