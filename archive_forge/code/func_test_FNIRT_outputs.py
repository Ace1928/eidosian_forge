from ..preprocess import FNIRT
def test_FNIRT_outputs():
    output_map = dict(field_file=dict(extensions=None), fieldcoeff_file=dict(extensions=None), jacobian_file=dict(extensions=None), log_file=dict(extensions=None), modulatedref_file=dict(extensions=None), out_intensitymap_file=dict(), warped_file=dict(extensions=None))
    outputs = FNIRT.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value