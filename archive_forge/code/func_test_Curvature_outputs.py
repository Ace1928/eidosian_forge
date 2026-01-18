from ..utils import Curvature
def test_Curvature_outputs():
    output_map = dict(out_gauss=dict(extensions=None), out_mean=dict(extensions=None))
    outputs = Curvature.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value