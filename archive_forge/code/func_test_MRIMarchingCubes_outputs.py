from ..utils import MRIMarchingCubes
def test_MRIMarchingCubes_outputs():
    output_map = dict(surface=dict(extensions=None))
    outputs = MRIMarchingCubes.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value