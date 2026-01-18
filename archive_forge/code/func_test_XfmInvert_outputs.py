from ..minc import XfmInvert
def test_XfmInvert_outputs():
    output_map = dict(output_file=dict(extensions=None), output_grid=dict(extensions=None))
    outputs = XfmInvert.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value