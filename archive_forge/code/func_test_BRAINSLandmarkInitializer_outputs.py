from ..brains import BRAINSLandmarkInitializer
def test_BRAINSLandmarkInitializer_outputs():
    output_map = dict(outputTransformFilename=dict(extensions=None))
    outputs = BRAINSLandmarkInitializer.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value