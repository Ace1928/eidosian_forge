from ..utils import SampleToSurface
def test_SampleToSurface_outputs():
    output_map = dict(hits_file=dict(extensions=None), out_file=dict(extensions=None), vox_file=dict(extensions=None))
    outputs = SampleToSurface.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value