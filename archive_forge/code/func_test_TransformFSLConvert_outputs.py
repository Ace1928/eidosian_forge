from ..utils import TransformFSLConvert
def test_TransformFSLConvert_outputs():
    output_map = dict(out_transform=dict(extensions=None))
    outputs = TransformFSLConvert.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value