from ..diffusion import DiffusionTensorScalarMeasurements
def test_DiffusionTensorScalarMeasurements_outputs():
    output_map = dict(outputScalar=dict(extensions=None, position=-1))
    outputs = DiffusionTensorScalarMeasurements.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value