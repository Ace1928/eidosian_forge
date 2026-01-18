from ..utils import VolumeMask
def test_VolumeMask_outputs():
    output_map = dict(lh_ribbon=dict(extensions=None), out_ribbon=dict(extensions=None), rh_ribbon=dict(extensions=None))
    outputs = VolumeMask.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value