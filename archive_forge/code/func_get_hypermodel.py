from keras_tuner.src import errors
from keras_tuner.src.api_export import keras_tuner_export
def get_hypermodel(hypermodel):
    """Gets a HyperModel from a HyperModel or callable."""
    if hypermodel is None:
        return None
    if isinstance(hypermodel, HyperModel):
        return hypermodel
    if not callable(hypermodel):
        raise errors.FatalValueError('The `hypermodel` argument should be either a callable with signature `build(hp)` returning a model, or an instance of `HyperModel`.')
    return DefaultHyperModel(hypermodel)