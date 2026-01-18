from ..brains import BRAINSSnapShotWriter
def test_BRAINSSnapShotWriter_outputs():
    output_map = dict(outputFilename=dict(extensions=None))
    outputs = BRAINSSnapShotWriter.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value