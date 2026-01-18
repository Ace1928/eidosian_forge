from ..utils import Sphere
def test_Sphere_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = Sphere.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value