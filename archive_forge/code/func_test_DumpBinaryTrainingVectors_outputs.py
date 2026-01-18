from ..featuredetection import DumpBinaryTrainingVectors
def test_DumpBinaryTrainingVectors_outputs():
    output_map = dict()
    outputs = DumpBinaryTrainingVectors.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value