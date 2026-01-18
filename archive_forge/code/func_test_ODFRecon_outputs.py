from ..odf import ODFRecon
def test_ODFRecon_outputs():
    output_map = dict(B0=dict(extensions=None), DWI=dict(extensions=None), ODF=dict(extensions=None), entropy=dict(extensions=None), max=dict(extensions=None))
    outputs = ODFRecon.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value