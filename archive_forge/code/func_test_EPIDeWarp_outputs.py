from ..epi import EPIDeWarp
def test_EPIDeWarp_outputs():
    output_map = dict(exf_mask=dict(extensions=None), exfdw=dict(extensions=None), unwarped_file=dict(extensions=None), vsm_file=dict(extensions=None))
    outputs = EPIDeWarp.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value