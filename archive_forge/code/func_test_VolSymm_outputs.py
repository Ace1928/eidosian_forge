from ..minc import VolSymm
def test_VolSymm_outputs():
    output_map = dict(output_file=dict(extensions=None), output_grid=dict(extensions=None), trans_file=dict(extensions=None))
    outputs = VolSymm.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value